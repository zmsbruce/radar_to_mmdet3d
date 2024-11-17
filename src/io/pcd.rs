use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
};

use anyhow::{anyhow, Context, Result};
use tracing::{debug, error, info, span, trace, warn, Level};

pub struct PcdReader;

#[derive(Debug)]
struct PcdHeader {
    fields: Vec<String>,
    size: Vec<usize>,
    data_type: Vec<char>,
    points: usize,
    data_format: String,
}

impl PcdReader {
    pub fn read_from_file(file_path: &str) -> Result<Vec<Vec<f64>>> {
        let span = span!(Level::TRACE, "PcdReader::read_from_file");
        let _enter = span.enter();

        info!("Opening file: {}", file_path);
        let file = File::open(file_path).map_err(|e| {
            error!("Failed to open file: {}: {}", file_path, e);
            e
        })?;
        let mut reader = BufReader::new(file);

        Self::read_from_reader(&mut reader)
    }

    fn read_from_reader<R: BufRead>(reader: &mut R) -> Result<Vec<Vec<f64>>> {
        let span = span!(Level::TRACE, "PcdReader::read_from_reader");
        let _enter = span.enter();

        trace!("Reading PCD header from reader...");
        let header = match Self::parse_pcd_header(reader) {
            Ok(header) => {
                debug!("Parsed PCD header: {:?}", header);
                header
            }
            Err(e) => {
                error!("Failed to parse PCD header: {}", e);
                return Err(e);
            }
        };

        trace!("Reading PCD data in {} format", header.data_format);
        if header.data_format == "ascii" {
            Self::read_pcd_ascii(reader, header)
        } else if header.data_format == "binary" {
            Self::read_pcd_binary(reader, header)
        } else {
            error!("Unsupported data format: {}", header.data_format);
            Err(anyhow!("Unsupported data format"))
        }
    }

    fn parse_pcd_header<R: BufRead>(reader: &mut R) -> Result<PcdHeader> {
        let span = span!(Level::TRACE, "PcdReader::parse_pcd_header");
        let _enter = span.enter();

        trace!("Parsing PCD header...");

        let mut fields = Vec::new();
        let mut size = Vec::new();
        let mut data_type = Vec::new();
        let mut points = 0;
        let mut data_format = String::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() || parts[0].starts_with('#') {
                continue;
            }

            match parts[0] {
                "FIELDS" => {
                    fields = parts[1..].iter().map(|s| s.to_string()).collect();
                    debug!("Parsed FIELDS: {:?}", fields);
                }
                "SIZE" => {
                    size = parts[1..]
                        .iter()
                        .map(|s| -> Result<_> {
                            s.parse::<usize>().context("Failed to parse SIZE")
                        })
                        .collect::<Result<Vec<usize>>>()?;
                    debug!("Parsed SIZE: {:?}", size);
                }
                "TYPE" => {
                    data_type = parts[1..]
                        .iter()
                        .map(|s| -> Result<_> {
                            s.chars()
                                .next()
                                .ok_or_else(|| anyhow!("Failed to parse TYPE"))
                        })
                        .collect::<Result<Vec<char>>>()?;
                    debug!("Parsed TYPE: {:?}", data_type);
                }
                "POINTS" => {
                    points = parts[1].parse::<usize>()?;
                    debug!("Parsed POINTS: {}", points);
                }
                "DATA" => {
                    data_format = parts[1].to_string();
                    debug!("Parsed DATA format: {}", data_format);
                    break;
                }
                _ => {
                    warn!("Unknown header field: {}", parts[0]);
                }
            }
        }

        Ok(PcdHeader {
            fields,
            size,
            data_type,
            points,
            data_format,
        })
    }

    fn read_pcd_binary<R: Read>(mut reader: R, header: PcdHeader) -> Result<Vec<Vec<f64>>> {
        let span = span!(Level::TRACE, "PcdReader::read_pcd_binary");
        let _enter = span.enter();

        trace!("Reading binary PCD data...");

        let mut points = Vec::with_capacity(header.points);

        let point_sz_bytes: usize = header.size.iter().sum();
        let mut buffer = vec![0u8; point_sz_bytes];

        for point_idx in 0..header.points {
            reader
                .read_exact(&mut buffer)
                .context(format!("Failed to read point {point_idx}"))?;

            let mut point = Vec::with_capacity(header.fields.len());
            let mut offset = 0;

            for (field_sz, field_type) in header.size.iter().zip(header.data_type.iter()) {
                let (start, end) = (offset, offset + field_sz);
                offset += field_sz;

                match field_type {
                    'F' => {
                        if *field_sz == 4 {
                            let value = f32::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to f32")?,
                            );
                            debug!("Parsed f32 value: {}", value);
                            point.push(value as f64);
                        } else if *field_sz == 8 {
                            let value = f64::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to f64")?,
                            );
                            debug!("Parsed f64 value: {}", value);
                            point.push(value);
                        } else {
                            return Err(anyhow!(
                                "Field size {field_sz} not matched to type {field_type}"
                            ));
                        }
                    }
                    'U' => {
                        let value = match *field_sz {
                            1 => buffer[start] as u64,
                            2 => u16::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to u16")?,
                            ) as u64,
                            4 => u32::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to u32")?,
                            ) as u64,
                            8 => u64::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to u64")?,
                            ),
                            _ => {
                                return Err(anyhow!(
                                    "Field size {field_sz} not matched to type {field_type}"
                                ))
                            }
                        };
                        debug!("Parsed unsigned value: {}", value);
                        point.push(value as f64);
                    }
                    'I' => {
                        let value = match *field_sz {
                            1 => buffer[start] as i64,
                            2 => i16::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to i16")?,
                            ) as i64,
                            4 => i32::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to i32")?,
                            ) as i64,
                            8 => i64::from_le_bytes(
                                buffer[start..end]
                                    .try_into()
                                    .context("Failed to parse buffer to i64")?,
                            ) as i64,
                            _ => {
                                return Err(anyhow!(
                                    "Field size {field_sz} not matched to type {field_type}"
                                ))
                            }
                        };
                        debug!("Parsed signed value: {}", value);
                        point.push(value as f64);
                    }
                    _ => {
                        return Err(anyhow!("Unsupported field type {field_type}"));
                    }
                }
            }

            debug!("Parsed point: {:?}", point);
            points.push(point);
        }

        info!(
            "Successfully parsed {} points from binary data",
            points.len()
        );
        Ok(points)
    }

    fn read_pcd_ascii<R: BufRead>(reader: R, header: PcdHeader) -> Result<Vec<Vec<f64>>> {
        let span = span!(Level::TRACE, "PcdReader::read_pcd_ascii");
        let _enter = span.enter();

        trace!("Reading ASCII PCD data...");
        let mut points = Vec::with_capacity(header.points);

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.context(format!("Failed to read line {}", line_idx))?;
            let values: Vec<&str> = line.split_whitespace().collect();

            let mut point = Vec::with_capacity(header.fields.len());

            for (value, (field_sz, field_type)) in values
                .iter()
                .zip(header.size.iter().zip(header.data_type.iter()))
            {
                match field_type {
                    'F' => {
                        let parsed_value = match *field_sz {
                            4 => value
                                .parse::<f32>()
                                .context(format!("Failed to parse '{}' as f32", value))?
                                as f64,
                            8 => value
                                .parse::<f64>()
                                .context(format!("Failed to parse '{}' as f64", value))?,
                            _ => {
                                return Err(anyhow!(
                                    "Unsupported float size: {} for field_type {}",
                                    field_sz,
                                    field_type
                                ));
                            }
                        };
                        debug!("Parsed float value: {}", parsed_value);
                        point.push(parsed_value);
                    }
                    'U' => {
                        let parsed_value = match *field_sz {
                            1 => value
                                .parse::<u8>()
                                .context(format!("Failed to parse '{}' as u8", value))?
                                as f64,
                            2 => value
                                .parse::<u16>()
                                .context(format!("Failed to parse '{}' as u16", value))?
                                as f64,
                            4 => value
                                .parse::<u32>()
                                .context(format!("Failed to parse '{}' as u32", value))?
                                as f64,
                            8 => value
                                .parse::<u64>()
                                .context(format!("Failed to parse '{}' as u64", value))?
                                as f64,
                            _ => {
                                return Err(anyhow!(
                                    "Unsupported unsigned integer size: {} for field_type {}",
                                    field_sz,
                                    field_type
                                ));
                            }
                        };
                        debug!("Parsed unsigned integer value: {}", parsed_value);
                        point.push(parsed_value);
                    }
                    'I' => {
                        let parsed_value = match *field_sz {
                            1 => value
                                .parse::<i8>()
                                .context(format!("Failed to parse '{}' as i8", value))?
                                as f64,
                            2 => value
                                .parse::<i16>()
                                .context(format!("Failed to parse '{}' as i16", value))?
                                as f64,
                            4 => value
                                .parse::<i32>()
                                .context(format!("Failed to parse '{}' as i32", value))?
                                as f64,
                            8 => value
                                .parse::<i64>()
                                .context(format!("Failed to parse '{}' as i64", value))?
                                as f64,
                            _ => {
                                return Err(anyhow!(
                                    "Unsupported signed integer size: {} for field_type {}",
                                    field_sz,
                                    field_type
                                ));
                            }
                        };
                        debug!("Parsed signed integer value: {}", parsed_value);
                        point.push(parsed_value as f64);
                    }
                    _ => {
                        return Err(anyhow!("Unsupported field type {}", field_type));
                    }
                }
            }

            debug!("Parsed point: {:?}", point);
            points.push(point);
        }

        info!(
            "Successfully parsed {} points from ASCII data",
            points.len()
        );
        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_pcd_ascii_data() -> String {
        r#"
# .PCD v0.7 - Point Cloud Data file format
FIELDS x y z
SIZE 4 4 4
TYPE F F F
POINTS 5
DATA ascii
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
10.0 11.0 12.0
13.0 14.0 15.0
"#
        .to_string()
    }

    fn create_pcd_binary_data() -> Vec<u8> {
        let header = r#"
# .PCD v0.7 - Point Cloud Data file format
FIELDS x y z
SIZE 4 4 4
TYPE F F F
POINTS 2
DATA binary
"#
        .to_string();

        let mut data = header.into_bytes();
        let points = vec![
            1.0_f32.to_le_bytes(),
            2.0_f32.to_le_bytes(),
            3.0_f32.to_le_bytes(),
            4.0_f32.to_le_bytes(),
            5.0_f32.to_le_bytes(),
            6.0_f32.to_le_bytes(),
        ];

        for bytes in points {
            data.extend_from_slice(&bytes);
        }

        data
    }

    #[test]
    fn test_parse_ascii_pcd() {
        let pcd_data = create_pcd_ascii_data();
        let mut cursor = Cursor::new(pcd_data);

        let result = PcdReader::read_from_reader(&mut cursor);
        assert!(result.is_ok());

        let points = result.unwrap();
        assert_eq!(points.len(), 5);
        assert_eq!(points[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(points[4], vec![13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_parse_binary_pcd() {
        let pcd_data = create_pcd_binary_data();
        let mut cursor = Cursor::new(pcd_data);

        let result = PcdReader::read_from_reader(&mut cursor);
        assert!(result.is_ok());

        let points = result.unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(points[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_invalid_pcd() {
        let invalid_pcd_data = "INVALID HEADER DATA".to_string();
        let mut cursor = Cursor::new(invalid_pcd_data);

        let result = PcdReader::read_from_reader(&mut cursor);
        assert!(result.is_err());
    }
}
