use std::{
    collections::VecDeque,
    ops::{Sub, SubAssign},
};

use nalgebra::{Point, RealField, Scalar};

pub fn dbscan<T, const D: usize>(points: &Vec<Point<T, D>>, eps: T, min_points: usize) -> Vec<isize>
where
    T: Scalar + Sub + SubAssign + RealField + Copy,
{
    let mut labels = vec![-1; points.len()];
    let mut cluster_id = 0;

    for i in 0..points.len() {
        if labels[i] != -1 {
            continue;
        }

        let neighbors = region_query(points, i, eps);
        if neighbors.len() < min_points {
            labels[i] = -2;
        } else {
            expand_cluster(
                points,
                &mut labels,
                i,
                neighbors,
                cluster_id,
                eps,
                min_points,
            );
            cluster_id += 1;
        }
    }

    labels
}

fn euclidean_distance<T, const D: usize>(p1: &Point<T, D>, p2: &Point<T, D>) -> T
where
    T: Scalar + Sub + SubAssign + RealField + Copy,
{
    (p1 - p2).norm()
}

fn expand_cluster<T, const D: usize>(
    points: &Vec<Point<T, D>>,
    labels: &mut Vec<isize>,
    point_idx: usize,
    neighbors: Vec<usize>,
    cluster_id: isize,
    eps: T,
    min_points: usize,
) where
    T: Scalar + Sub + SubAssign + RealField + Copy,
{
    labels[point_idx] = cluster_id;

    let mut queue = VecDeque::from(neighbors);

    while let Some(current_idx) = queue.pop_front() {
        if labels[current_idx] == -2 {
            labels[current_idx] = cluster_id;
        }

        if labels[current_idx] != -1 {
            continue;
        }

        labels[current_idx] = cluster_id;

        let current_neighbors = region_query(points, current_idx, eps);
        if current_neighbors.len() >= min_points {
            queue.extend(current_neighbors);
        }
    }
}

fn region_query<T, const D: usize>(
    points: &Vec<Point<T, D>>,
    point_idx: usize,
    eps: T,
) -> Vec<usize>
where
    T: Scalar + Sub + SubAssign + RealField + Copy,
{
    let mut neighbors = Vec::new();

    for (i, point) in points.iter().enumerate() {
        if euclidean_distance(&points[point_idx], point) <= eps {
            neighbors.push(i);
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3};

    use super::*;

    #[test]
    fn test_dbscan_clustering() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(1.0, 1.0),   // Cluster 0
            Point2::new(1.1, 1.1),   // Cluster 0
            Point2::new(0.9, 1.0),   // Cluster 0
            Point2::new(10.0, 10.0), // Cluster 1
            Point2::new(10.1, 10.1), // Cluster 1
            Point2::new(9.9, 10.0),  // Cluster 1
            Point2::new(50.0, 50.0), // Noise
        ];

        let eps = 0.5;
        let min_points = 2;

        let labels = dbscan(&points, eps, min_points);

        assert_eq!(labels.len(), points.len());

        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);

        assert_eq!(labels[3], 1);
        assert_eq!(labels[4], 1);
        assert_eq!(labels[5], 1);

        assert_eq!(labels[6], -2);
    }

    #[test]
    fn test_dbscan_with_no_clusters() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 3.0),
        ];

        let eps = 0.5;
        let min_points = 2;

        let labels = dbscan(&points, eps, min_points);

        for label in labels {
            assert_eq!(label, -2);
        }
    }

    #[test]
    fn test_dbscan_single_cluster() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(1.0, 1.0),
            Point2::new(1.1, 1.1),
            Point2::new(0.9, 1.0),
        ];

        let eps = 0.5;
        let min_points = 2;

        let labels = dbscan(&points, eps, min_points);

        let cluster_id = labels[0];
        assert!(cluster_id >= 0);

        for label in labels {
            assert_eq!(label, cluster_id);
        }
    }

    #[test]
    fn test_dbscan_clustering_3d() {
        let points: Vec<Point3<f64>> = vec![
            Point3::new(1.0, 1.0, 1.0),    // Cluster 0
            Point3::new(1.1, 1.1, 1.1),    // Cluster 0
            Point3::new(0.9, 1.0, 1.0),    // Cluster 0
            Point3::new(10.0, 10.0, 10.0), // Cluster 1
            Point3::new(10.1, 10.1, 10.1), // Cluster 1
            Point3::new(9.9, 10.0, 10.0),  // Cluster 1
            Point3::new(50.0, 50.0, 50.0), // Noise
        ];

        let eps = 0.5;
        let min_points = 2;

        let labels = dbscan(&points, eps, min_points);

        assert_eq!(labels.len(), points.len());

        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);

        assert_eq!(labels[3], 1);
        assert_eq!(labels[4], 1);
        assert_eq!(labels[5], 1);

        assert_eq!(labels[6], -2);
    }
}
