import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


class GameLogger:
    """游戏日志管理器"""
    
    def __init__(self, log_dir: str = "./logs"):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志存储目录
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置根日志
        self._configure_root_logger()
        
        # 创建游戏日志目录
        self.game_log_dir = f"{self.log_dir}/games/"
        self.service_log_dir = f"{self.log_dir}/services/"
        os.makedirs(self.game_log_dir, exist_ok=True)
        os.makedirs(self.service_log_dir, exist_ok=True)
    
    def _configure_root_logger(self):
        """
        配置根日志器
        """
        # 设置根日志级别
        logging.root.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        today = datetime.now().strftime("%Y%m%d")
        file_handler = RotatingFileHandler(
            filename=f"{self.log_dir}/game_system_{today}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到根日志器
        logging.root.addHandler(console_handler)
        logging.root.addHandler(file_handler)
    
    def get_game_logger(self, game_id: str, component: str) -> logging.Logger:
        """
        获取游戏特定的日志器
        
        Args:
            game_id: 游戏ID
            component: 组件名称
            
        Returns:
            logging.Logger实例
        """
        logger_name = f"game.{game_id}.{component}"
        logger = logging.getLogger(logger_name)
        
        # 为游戏日志创建单独的文件处理器
        game_log_file = f"{self.game_log_dir}/{game_id}.log"
        file_handler = RotatingFileHandler(
            filename=game_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志器
        logger.addHandler(file_handler)
        
        return logger
    
    def get_service_logger(self, service_name: str) -> logging.Logger:
        """
        获取服务日志器
        
        Args:
            service_name: 服务名称
            
        Returns:
            logging.Logger实例
        """
        logger_name = f"service.{service_name}"
        logger = logging.getLogger(logger_name)
        
        # 为服务日志创建单独的文件处理器
        service_log_file = f"{self.service_log_dir}/{service_name}.log"
        file_handler = RotatingFileHandler(
            filename=service_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志器
        logger.addHandler(file_handler)
        
        return logger
    
    def clean_old_logs(self, days_to_keep: int = 7):
        """
        清理旧日志文件
        
        Args:
            days_to_keep: 保留天数
        """
        import time
        import glob
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        # 清理根日志目录
        for log_file in glob.glob(f"{self.log_dir}/*.log"):
            if os.path.getmtime(log_file) < cutoff_time:
                os.remove(log_file)
        
        # 清理游戏日志目录
        for log_file in glob.glob(f"{self.game_log_dir}/*.log"):
            if os.path.getmtime(log_file) < cutoff_time:
                os.remove(log_file)
        
        # 清理服务日志目录
        for log_file in glob.glob(f"{self.service_log_dir}/*.log"):
            if os.path.getmtime(log_file) < cutoff_time:
                os.remove(log_file)


# 创建全局日志管理器实例
game_logger = GameLogger()