from controller.cli_arguments import CLIArguments

from utils.logger import Logger


def main() -> int:
    logger: Logger = Logger(class_name=__name__)

    args: CLIArguments = CLIArguments()
    args.parse()
    model_name = args.get("model_name")

    try:

        logger.info(f"Hello from {__name__}")
        logger.info(f"Model Name: {model_name}")

    except Exception as e:
        raise Exception(f"Error Message: {e}")

    return 0


if __name__ == "__main__":
    main()
