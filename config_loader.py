"""
設定ファイル読み込みモジュール
config.yamlを読み込んで、アプリケーション設定を提供します。
"""

import os
import yaml
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class ConfigLoader:
    """設定ファイルを読み込むクラス"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス（指定しない場合はデフォルトのconfig.yamlを使用）
        """
        if config_path is None:
            # スクリプトと同じディレクトリのconfig.yamlを使用
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")

    def get(self, *keys, default=None) -> Any:
        """
        ネストされた設定値を取得

        Args:
            *keys: 設定キー（ネスト可能）
            default: デフォルト値

        Returns:
            設定値、見つからない場合はdefault

        Example:
            config.get('camera', 'network', 'video_url')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_camera_type(self) -> str:
        """カメラタイプを取得"""
        return self.get("camera", "type", default="network")

    def get_network_video_url(self) -> str:
        """ネットワークカメラのビデオURLを取得"""
        return self.get(
            "camera",
            "network",
            "video_url",
            default="http://192.168.110.102:8080/video",
        )

    def get_network_photo_url(self) -> str:
        """ネットワークカメラの静止画URLを取得"""
        return self.get(
            "camera",
            "network",
            "photo_url",
            default="http://192.168.110.102:8080/photoaf.jpg",
        )

    def get_network_retry_count(self) -> int:
        """ネットワークカメラの接続リトライ回数を取得"""
        return self.get("camera", "network", "retry_count", default=3)

    def get_local_device_index(self) -> int:
        """ローカルカメラのデバイスインデックスを取得"""
        return self.get("camera", "local", "device_index", default=0)

    def get_buffer_size(self) -> int:
        """カメラバッファサイズを取得"""
        return self.get("camera", "buffer_size", default=1)

    def get_frame_interval_ms(self) -> int:
        """フレーム更新間隔（ミリ秒）を取得"""
        return self.get("camera", "frame_interval_ms", default=30)

    def get_aruco_dict_type(self) -> str:
        """ArUco辞書タイプを取得"""
        return self.get("aruco", "dict_type", default="DICT_4X4_50")

    def get_aruco_area_ratio_threshold(self) -> float:
        """ArUcoマーカーの面積比率閾値を取得"""
        return self.get("aruco", "area_ratio_threshold", default=0.001)

    def get_aruco_fill_threshold(self) -> float:
        """ArUcoマーカーの充填率閾値を取得"""
        return self.get("aruco", "fill_threshold", default=0.6)

    def get_auto_capture_delay_ms(self) -> int:
        """自動撮影の遅延時間（ミリ秒）を取得"""
        return self.get("aruco", "auto_capture_delay_ms", default=2000)

    def get_capture_cooldown_ms(self) -> int:
        """撮影後のクールダウン時間（ミリ秒）を取得"""
        return self.get("aruco", "capture_cooldown_ms", default=3000)

    def get_gemini_api_key(self) -> str:
        """Gemini APIキーを取得"""
        # 環境変数を優先、なければ設定ファイルから
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            return env_key
        return self.get("llm", "api_key", default="")

    def get_aruco_marker_size_mm(self) -> int:
        """ArUcoマーカーのサイズ（mm）を取得"""
        return self.get("aruco", "marker_size_mm", default=80)

    def get_aruco_output_dpi(self) -> int:
        """台形補正の出力DPIを取得"""
        return self.get("aruco", "output_dpi", default=200)

    def get_white_balance_enabled_by_default(self) -> bool:
        """ホワイトバランス補正のデフォルト状態を取得"""
        return self.get(
            "image_processing", "white_balance", "enabled_by_default", default=True
        )

    def get_gaussian_blur_kernel(self) -> list:
        """ガウシアンブラーのカーネルサイズを取得"""
        return self.get(
            "image_processing", "edge_detection", "gaussian_blur_kernel", default=[5, 5]
        )

    def get_canny_threshold1(self) -> int:
        """Cannyエッジ検出の閾値1を取得"""
        return self.get(
            "image_processing", "edge_detection", "canny_threshold1", default=50
        )

    def get_canny_threshold2(self) -> int:
        """Cannyエッジ検出の閾値2を取得"""
        return self.get(
            "image_processing", "edge_detection", "canny_threshold2", default=150
        )

    def get_hough_threshold(self) -> int:
        """ハフ変換の閾値を取得"""
        return self.get("image_processing", "hough_transform", "threshold", default=160)

    def get_hough_min_line_length(self) -> int:
        """ハフ変換の最小直線長を取得"""
        return self.get(
            "image_processing", "hough_transform", "min_line_length", default=240
        )

    def get_hough_max_line_gap(self) -> int:
        """ハフ変換の最大直線ギャップを取得"""
        return self.get(
            "image_processing", "hough_transform", "max_line_gap", default=30
        )

    def get_window_width(self) -> int:
        """ウィンドウ幅を取得"""
        return self.get("ui", "window", "width", default=1200)

    def get_window_height(self) -> int:
        """ウィンドウ高さを取得"""
        return self.get("ui", "window", "height", default=800)

    def get_video_label_min_width(self) -> int:
        """ビデオラベルの最小幅を取得"""
        return self.get("ui", "video_label", "min_width", default=640)

    def get_video_label_min_height(self) -> int:
        """ビデオラベルの最小高さを取得"""
        return self.get("ui", "video_label", "min_height", default=480)

    def get_ocr_output_max_height(self) -> int:
        """OCR出力エリアの最大高さを取得"""
        return self.get("ui", "ocr_output", "max_height", default=150)

    def get_captures_dir(self) -> str:
        """キャプチャ保存ディレクトリを取得"""
        return self.get("directories", "captures", default="captures")

    def get_subject_mappings_file(self) -> str:
        """教科マッピングファイル名を取得"""
        return self.get("files", "subject_mappings", default="subject_mappings.json")

    def get_camera_params_file(self) -> str:
        """カメラパラメータファイル名を取得"""
        return self.get("files", "camera_params", default="camera_params.json")


# グローバルインスタンス（アプリケーション起動時に一度だけ読み込む）
_config_instance = None


def get_config() -> ConfigLoader:
    """設定のグローバルインスタンスを取得"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance


def reload_config():
    """設定を再読み込み"""
    global _config_instance
    _config_instance = ConfigLoader()
    return _config_instance
