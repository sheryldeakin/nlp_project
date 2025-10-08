from nlp_project.utils.logger import Logger


def main() -> int:
    logger: Logger = Logger(class_name=__name__)

    logger.info(f"Hello from {__name__}")
    logger.warning(f"Hello from {__name__}")
    logger.error(f"Hello from {__name__}")

    return 0


if __name__ == "__main__":
    main()
