from classifiers.classifier_factory import ClassifierFactory


class TrainingInterface:
    def start(self):
        self._print_header()
        option = self._prompt_for_classifier_type()
        self._handle_option(option)
        return None

    def _print_header(self):
        print(
            "Welcome to the training UI. We need to ask some questions before we continue..."
        )

    def _prompt_for_classifier_type(self):
        print(
            "Would you like to train a (B)enchmark classifier or a (S)emi-Supervised classifier?\n"
            "Type (?) for an explanation on classifier options, or anything else to cancel."
        )
        return self._prompt_user()


    def _prompt_user(self):
        return input("ui/train > ")

    def _handle_option(self, option):
        option_lower = option.lower()
        if option_lower == "b":
            target_file_name = self._prompt_for_target_file_name()
            data_directory = self._prompt_for_data_directory()
            return self._create_benchmark_classifier(target_file_name, data_directory)
        elif option_lower == "s":
            pass
        elif option_lower == "?":
            self._print_classifier_details()
            self._prompt_for_classifier_type()
        else:
            print("Command \"{}\" not supported. Exiting training UI.".format(option))

    def _print_classifier_details(self):
        print(
            "Sorry! Not supported yet. Google it instead."
        )

    def _prompt_for_target_file_name(self):
        print(
            "Please provide target file. This is a CSV file that should be formatted as follows...\n"
            "   [filename to record text],[record label]"
        )
        return self._prompt_user()

    def _prompt_for_data_directory(self):
        print(
            "Please provide the directory where the record text files reside.\n"
            "   If files exist in the directory, but are not included in the target file, they will\n"
            "   not be included when training."
        )
        return self._prompt_user()

    def _create_benchmark_classifier(self, target_file_name, data_directory):
        return ClassifierFactory().build_benchmark_classifier(target_file_name, data_directory)


