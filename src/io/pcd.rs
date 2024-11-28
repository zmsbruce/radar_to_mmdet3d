use anyhow::{anyhow, Result};
use nalgebra::Point3;
use std::fmt::Write;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write as StdWrite};
use std::path::Path;
use tracing::{debug, error, span, trace, warn, Level};

struct PcdHeader {
    fields: Vec<String>,
    size: Vec<usize>,
    data_type: Vec<char>,
    points: usize,
    data_format: String,
}

pub fn read_pcd_from_file<P>(file_path: P) -> Result<Vec<Vec<f64>>>
where
    P: AsRef<std::path::Path> + std::fmt::Debug,
{
    let span = span!(Level::TRACE, "read_pcd_from_file");
    let _enter = span.enter();

    debug!("Opening file: {:?}", file_path);
    let file = File::open(&file_path).map_err(|e| {
        error!("Failed to open file: {:?}: {}", file_path, e);
        e
    })?;
    let mut reader = BufReader::new(file);

    let points = read_pcd_from_reader(&mut reader)?;

    trace!("Successfully read points with length: {}", points.len());
    Ok(points)
}

fn read_pcd_from_reader<R: BufRead>(reader: &mut R) -> Result<Vec<Vec<f64>>> {
    let span = span!(Level::TRACE, "read_pcd_from_reader");
    let _enter = span.enter();

    trace!("Reading PCD header from reader...");
    let header = parse_pcd_header(reader)?;

    trace!("Reading PCD data in {} format", header.data_format);
    if header.data_format == "ascii" {
        read_pcd_ascii(reader, &header)
    } else if header.data_format == "binary" {
        read_pcd_binary(reader, &header)
    } else {
        error!("Unsupported data format: {}", header.data_format);
        Err(anyhow!("Unsupported data format"))
    }
}

fn parse_pcd_header<R: BufRead>(reader: &mut R) -> Result<PcdHeader> {
    let span = span!(Level::TRACE, "parse_pcd_header");
    let _enter = span.enter();

    trace!("Parsing PCD header...");

    let mut fields = Vec::new();
    let mut size = Vec::new();
    let mut data_type = Vec::new();
    let mut points = 0;
    let mut data_format = String::new();

    let lines = reader.lines();
    for line in lines {
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
                    .map(|s| -> Result<_, _> { s.parse::<usize>() })
                    .collect::<Result<Vec<usize>, _>>()?;
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

fn read_pcd_binary<R: Read>(mut reader: R, header: &PcdHeader) -> Result<Vec<Vec<f64>>> {
    let span = span!(Level::TRACE, "read_pcd_binary");
    let _enter = span.enter();

    trace!("Reading binary PCD data...");

    let mut points = Vec::with_capacity(header.points);

    let point_sz_bytes: usize = header.size.iter().sum();
    let mut buffer = vec![0u8; point_sz_bytes];

    for _ in 0..header.points {
        reader.read_exact(&mut buffer)?;

        let mut point = Vec::with_capacity(header.fields.len());
        let mut offset = 0;

        for (field_sz, field_type) in header.size.iter().zip(header.data_type.iter()) {
            let (start, end) = (offset, offset + field_sz);
            offset += field_sz;

            match field_type {
                'F' => {
                    if *field_sz == 4 {
                        let value = f32::from_le_bytes(buffer[start..end].try_into()?);
                        debug!("Parsed f32 value: {}", value);
                        point.push(value as f64);
                    } else if *field_sz == 8 {
                        let value = f64::from_le_bytes(buffer[start..end].try_into()?);
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
                        2 => u16::from_le_bytes(buffer[start..end].try_into()?) as u64,
                        4 => u32::from_le_bytes(buffer[start..end].try_into()?) as u64,
                        8 => u64::from_le_bytes(buffer[start..end].try_into()?),
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
                        2 => i16::from_le_bytes(buffer[start..end].try_into()?) as i64,
                        4 => i32::from_le_bytes(buffer[start..end].try_into()?) as i64,
                        8 => i64::from_le_bytes(buffer[start..end].try_into()?) as i64,
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

    debug!(
        "Successfully parsed {} points from binary data",
        points.len()
    );
    Ok(points)
}

fn read_pcd_ascii<R: BufRead>(reader: R, header: &PcdHeader) -> Result<Vec<Vec<f64>>> {
    let span = span!(Level::TRACE, "read_pcd_ascii");
    let _enter = span.enter();

    trace!("Reading ASCII PCD data...");
    let mut points = Vec::with_capacity(header.points);

    let lines = reader.lines();
    for line in lines {
        let line = line?;
        let values: Vec<&str> = line.split_whitespace().collect();

        let mut point = Vec::with_capacity(header.fields.len());

        for (value, (field_sz, field_type)) in values
            .iter()
            .zip(header.size.iter().zip(header.data_type.iter()))
        {
            match field_type {
                'F' => {
                    let parsed_value = match *field_sz {
                        4 => value.parse::<f32>()? as f64,
                        8 => value.parse::<f64>()?,
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
                        1 => value.parse::<u8>()? as f64,
                        2 => value.parse::<u16>()? as f64,
                        4 => value.parse::<u32>()? as f64,
                        8 => value.parse::<u64>()? as f64,
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
                        1 => value.parse::<i8>()? as f64,
                        2 => value.parse::<i16>()? as f64,
                        4 => value.parse::<i32>()? as f64,
                        8 => value.parse::<i64>()? as f64,
                        _ => {
                            return Err(anyhow!(
                                "Unsupported signed integer size: {} for field_type {}",
                                field_sz,
                                field_type
                            ));
                        }
                    };
                    debug!("Parsed signed integer value: {}", parsed_value);
                    point.push(parsed_value);
                }
                _ => {
                    return Err(anyhow!("Unsupported field type {}", field_type));
                }
            }
        }

        debug!("Parsed point: {:?}", point);
        points.push(point);
    }

    debug!(
        "Successfully parsed {} points from ASCII data",
        points.len()
    );
    Ok(points)
}

pub fn save_pointcloud<P>(points: &[Point3<f32>], path: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let file_path = path.as_ref();
    let file = File::create(file_path)?;

    let mut writer = BufWriter::new(file);

    writer.write_all(b"VERSION .7\n")?;
    writer.write_all(b"FIELDS x y z\n")?;
    writer.write_all(b"SIZE 4 4 4\n")?;
    writer.write_all(b"TYPE F F F\n")?;
    writer.write_all(b"COUNT 1 1 1\n")?;

    writer.write_all(format!("WIDTH {}\n", points.len()).as_bytes())?;
    writer.write_all(b"HEIGHT 1\n")?;
    writer.write_all(b"VIEWPOINT 0 0 0 1 0 0 0\n")?;
    writer.write_all(format!("POINTS {}\n", points.len()).as_bytes())?;
    writer.write_all(b"DATA ascii\n")?;

    let mut buffer = String::with_capacity(points.len() * 32);
    for point in points {
        writeln!(&mut buffer, "{} {} {}", point.x, point.y, point.z)?;
    }

    writer.write_all(buffer.as_bytes())?;
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::{Cursor, Write},
    };
    use tempfile::NamedTempFile;

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

        let result = read_pcd_from_reader(&mut cursor);
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

        let result = read_pcd_from_reader(&mut cursor);
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

        let result = read_pcd_from_reader(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_pcd_from_file() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create tempfile");
        let pcd_data = create_pcd_ascii_data();

        temp_file
            .write_all(pcd_data.as_bytes())
            .expect("Failed to write PCD data to tempfile");
        let file_path = temp_file.path().to_str().unwrap().to_string();

        let points = read_pcd_from_file(&file_path);
        assert!(points.is_ok());
        let points = points.unwrap();
        assert_eq!(points.len(), 5);
        assert_eq!(points[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(points[4], vec![13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_save_pointcloud_() {
        let points = vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(4.0, 5.0, 6.0),
            Point3::new(7.0, 8.0, 9.0),
        ];
        let temp_file = tempfile::NamedTempFile::new().expect("Failed to create tempfile");
        let file_path = temp_file.path().to_path_buf();

        save_pointcloud(&points, &file_path).expect("Failed to save point cloud");

        let saved_content =
            fs::read_to_string(&file_path).expect("Failed to read saved point cloud file");

        assert!(saved_content.contains("VERSION .7"));
        assert!(saved_content.contains("FIELDS x y z"));
        assert!(saved_content.contains("SIZE 4 4 4"));
        assert!(saved_content.contains("TYPE F F F"));
        assert!(saved_content.contains("COUNT 1 1 1"));
        assert!(saved_content.contains("WIDTH 3"));
        assert!(saved_content.contains("HEIGHT 1"));
        assert!(saved_content.contains("VIEWPOINT 0 0 0 1 0 0 0"));
        assert!(saved_content.contains("POINTS 3"));
        assert!(saved_content.contains("DATA ascii"));

        for point in points {
            let point_str = format!("{} {} {}", point.x, point.y, point.z);
            assert!(saved_content.contains(&point_str));
        }
    }
}
