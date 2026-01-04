from openai import OpenAI


class LabelWordExtension:
    def __init__(self, initial_label_words, prompt, model, api_key, output_path):
        self.initial_label_words = initial_label_words
        self.prompt = prompt
        self.model = model
        self.output_path = output_path
        self.client = OpenAI(api_key=api_key)

    def _construct_user_prompt(self):
        class_names = self.initial_label_words.values()
        user_prompt = "نام کلاس ها:\n"
        user_prompt += "\n".join(class_names)
        self.user_prompt = user_prompt

    def gpt_response_to_label_words(self, response):
        response_per_classes = response.split("\n")
        result = {k: [v] for k, v in self.initial_label_words.items()}
        all_keys, all_initial_values = list(self.initial_label_words.keys()), list(self.initial_label_words.values())
        for index, rpc in enumerate(response_per_classes):
            _, value = rpc.split(":")
            value = value.strip().split("-")
            value = [v.strip() for v in value]
            result[all_keys[index]] += value
        return result

    def write_label_words(self, new_label_words):
        output_file = open(self.output_path, "w")
        file_content = ""
        for key, value in new_label_words.items():
            file_content += ",".join(value)
            file_content += "\n"
        file_content = file_content.strip()
        output_file.write(file_content)

    def extend_label_words(self):
        self._construct_user_prompt()
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        response = response.choices[0].message.content
        new_label_words = self.gpt_response_to_label_words(response)
        self.write_label_words(new_label_words)
        return new_label_words
