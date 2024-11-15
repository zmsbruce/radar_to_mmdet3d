use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct DetectorConfig {
    pub car_onnx_path: String,
    pub armor_onnx_path: String,
    pub car_conf_thresh: f32,
    pub armor_conf_thresh: f32,
    pub car_nms_thresh: f32,
    pub armor_nms_thresh: f32,
    pub execution: String,
}
