from controller.cli_arguments import CLIArguments
from controller.controller import Controller
from utils.logger import Logger


def main() -> int:
    logger: Logger = Logger(class_name=__name__)

    args: CLIArguments = CLIArguments()
    args.parse()
    model_name_str: str = args.get("model_name")

    try:

        controller: Controller = Controller()

        logger.info(f"Model Name: {model_name_str}")

        controller.execute_model(model_name_str=model_name_str)



    except Exception as e:
        raise Exception(f"Error Message: {e}")

    return 0


if __name__ == "__main__":
    main()
