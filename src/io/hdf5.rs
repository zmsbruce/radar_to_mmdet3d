use anyhow::{anyhow, Context, Result};
use hdf5::{Dataset, File};
use nalgebra::Point3;
use ndarray_015::{s, Ix2};
use tracing::error;

pub struct Hdf5PointCloudReader {
    dataset: Dataset,
    pub filename: String,
}

impl Hdf5PointCloudReader {
    pub fn from_file<P>(file_path: P) -> Result<Self>
    where
        P: AsRef<std::path::Path> + std::fmt::Debug + std::marker::Copy,
    {
        let file = File::open(file_path)
            .context("Failed to open hdf5 file")
            .map_err(|e| {
                error!("Failed to open hdf5 file {:?}: {}", file_path, e);
                e
            })?;

        let dataset = file
            .datasets()
            .context("Failed to get datasets for hdf5 file")
            .map_err(|e| {
                error!("Failed to get datasets: {}", e);
                e
            })?
            .into_iter()
            .find(|dataset| {
                let shape = dataset.shape();
                shape.len() == 3 && shape[2] == 3
            })
            .ok_or_else(|| anyhow!("Pointcloud sequence data must have shape [F, N, 3]"))?;

        Ok(Self {
            dataset,
            filename: file_path
                .as_ref()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
        })
    }

    #[inline]
    pub fn get_frame_num(&self) -> usize {
        self.dataset.shape()[0]
    }

    pub fn read_pointcloud_frame(&self, frame_idx: usize) -> Result<Vec<Point3<f32>>> {
        let num_frames = self.get_frame_num();
        if frame_idx >= num_frames {
            error!("Frame index {frame_idx} out of range {}", num_frames);
            return Err(anyhow!("Frame index out of range"));
        }

        let slice = s![frame_idx, .., ..];

        let dtype = self
            .dataset
            .dtype()
            .context("Failed to get dataset element type")?;

        if dtype.is::<f32>() {
            let points = self
                .dataset
                .read_slice::<f32, _, Ix2>(slice)?
                .as_slice()
                .ok_or_else(|| anyhow!("Failed to convert ndarray to slice"))?
                .chunks(3)
                .map(|chunk| Point3::new(chunk[0], chunk[1], chunk[2])) // 转换为 f32
                .collect();

            Ok(points)
        } else if dtype.is::<f64>() {
            let points = self
                .dataset
                .read_slice::<f64, _, Ix2>(slice)?
                .as_slice()
                .ok_or_else(|| anyhow!("Failed to convert ndarray to slice"))?
                .chunks(3)
                .map(|chunk| Point3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32)) // 转换为 f32
                .collect();

            Ok(points)
        } else {
            Err(anyhow!(
                "Unsupported dataset element type: {:?}",
                self.dataset.dtype()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_pointcloud_sequence() -> Result<()> {
        let file_path = "assets/test/test_point_clouds.h5";
        let reader = Hdf5PointCloudReader::from_file(file_path)?;

        let frame_num = reader.get_frame_num();

        let num_frames = 2;
        let num_points_per_frame = 5;
        assert_eq!(frame_num, num_frames, "Number of frames mismatch");

        for frame_idx in 0..num_frames {
            let frame = reader.read_pointcloud_frame(frame_idx)?;
            assert_eq!(
                frame.len(),
                num_points_per_frame,
                "Number of points in frame {} mismatch",
                frame_idx
            );

            for (point_idx, point) in frame.iter().enumerate() {
                let expected_x = frame_idx as f32 + point_idx as f32 * 0.1;
                let expected_y = frame_idx as f32 + point_idx as f32 * 0.2;
                let expected_z = frame_idx as f32 + point_idx as f32 * 0.3;

                assert!(
                    (point.x - expected_x).abs() < 1e-6,
                    "Point x mismatch in frame {}, point {}",
                    frame_idx,
                    point_idx
                );
                assert!(
                    (point.y - expected_y).abs() < 1e-6,
                    "Point y mismatch in frame {}, point {}",
                    frame_idx,
                    point_idx
                );
                assert!(
                    (point.z - expected_z).abs() < 1e-6,
                    "Point z mismatch in frame {}, point {}",
                    frame_idx,
                    point_idx
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_invalid_dataset_shape() -> Result<()> {
        let temp_file = NamedTempFile::new().context("Failed to create temporary file")?;
        let file_path = temp_file.path();

        let file = File::create(file_path).context("Failed to create test HDF5 file")?;
        let shape = [2, 5, 2]; // Invalid shape: last dimension must be 3
        file.new_dataset::<f32>()
            .shape(shape)
            .create("point_clouds")?;

        let reader = Hdf5PointCloudReader::from_file(file_path);
        assert!(reader.is_err(), "Expected error for invalid dataset shape");

        Ok(())
    }

    #[test]
    fn test_missing_point_clouds_dataset() -> Result<()> {
        let temp_file = NamedTempFile::new().context("Failed to create temporary file")?;
        let file_path = temp_file.path();

        File::create(file_path).context("Failed to create test HDF5 file")?;

        let reader = Hdf5PointCloudReader::from_file(file_path);
        assert!(
            reader.is_err(),
            "Expected error for missing point_clouds dataset"
        );

        Ok(())
    }
}
