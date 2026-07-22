import logging
import sys
from pathlib import Path


def setup_logger(name: str = "app_logger", log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """Khởi tạo và cấu hình logger dùng chung cho toàn bộ dự án."""

    # Cấu hình stdout/stderr sử dụng UTF-8 trên Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")# Thực hiện ghi log chữ Tiếng Việt
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Trường hợp môi trường không hỗ trợ reconfigure()
        pass

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Không cho log truyền lên root logger/Hydra
    # Tránh việc một log bị in ra 2 lần
    logger.propagate = False

    # Tránh tạo nhiều handler nếu setup_logger() được gọi nhiều lần
    if logger.handlers:
        return logger

    # Format log
    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] - "
            "(%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler xuất log ra Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Handler ghi log ra File
    try:
        log_path = Path(log_file)

        # Tạo thư mục nếu chưa tồn tại
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            filename=log_path,
            mode="a",
            encoding="utf-8",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    except Exception as e:
        # Dùng print thay vì logger.error để tránh lỗi handler
        print(
            f"Không thể tạo file log tại {log_file}: {e}",
            file=sys.stderr,
        )

    return logger

# Tạo logger mặc định
logger = setup_logger()