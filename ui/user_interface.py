from ui.training_interface import TrainingInterface


class UserInterface:
    def __init__(self):
        self.quit = False
        self.classifiers = []

    def start(self):
        self._print_header()
        while not self.quit:
            command = self._prompt_user()
            self._handle_command(command)

    def _print_header(self):
        print(
            "==========================================================================\n"
            "=   Welcome to Joel's Semi-Supervised Text Classification Program        =\n"
            "=      Type '?' or 'help' to view available commands.                    =\n"
            "=========================================================================="
        )

    def _prompt_user(self):
        return input("ui > ")

    def _handle_command(self, command):
        command_lower = command.lower()
        if command_lower == "t" or command_lower == "train":
            self._train()
        elif command_lower == "s" or command_lower == "show":
            self._show_classifiers()
        elif command_lower == "q" or command_lower == "quit":
            self._quit()
        elif command_lower == "?" or command_lower == "help":
            self._print_commands()
        else:
            print("Command \"{}\" not supported, going back.".format(command))

    def _print_commands(self):
        print("(T)rain a new classifier.\n"
              "(S)how trained classifiers.")

    def _show_classifiers(self):
        for classifier in self.classifiers:
            print("foobar.")

    def _train(self):
        ti = TrainingInterface()
        self.classifiers.append(ti.start())

    def _quit(self):
        self.quit = True
        print("Bye!")


def main():
    ui = UserInterface()
    ui.start()


if __name__ == "__main__":
    main()
