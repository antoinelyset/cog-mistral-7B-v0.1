# teknium/OpenHermes-2-Mistral-7B Cog

## About

This is a very basic implementation of [teknium/OpenHermes-2-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2-Mistral-7B) as a Cog model.

- OpenAI `messages` API supported as JSON `prompt`, thanks to `tokenizer.apply_chat_template` so it should work with any `tokenizer_config.json` with a `"chat_template"`, [example](https://huggingface.co/teknium/OpenHermes-2-Mistral-7B/blob/main/tokenizer_config.json#L52).
- Tensorizer to convert the pre-downloaded weights, and speed the boot time. You will need quite a bit of RAM (around 2 times the model size, so here ~26GB). It's also possible to download them at runtime, you will need to update `predict.py`.
- Simple inference engine with `transformers`
- You can swap the `MODEL_NAMEL` and change the `name` in `cog.yaml` and it should work with a lot of different Hugging Face models.

## Usage

You will need to install [Cog](https://github.com/replicate/cog/blob/main/docs/getting-started.md#install-cog)

Then download the pre-trained weights:

    cog run script/download-weights

Then build the container:

    cog build --separate-weights

Finally, run the container and test it:

    docker run --gpus all -p 9090:8080 r8.im/antoinelyset/openhermes-2-mistral-7b
    curl -X POST -H "Content-Type: application/json" -d '{"input": {"prompt": "[{\"role\": \"system\",\"content\":\"You are a helpful assistant.\"},{\"role\": \"user\",\"content\": \"What is Slite?\"}]"}}' http://localhost:9090/predictions

You can push it to Replicate, first create a model [https://replicate.com/create](https://replicate.com/create) and then:

    cog login
    docker push r8.im/antoinelyset/openhermes-2-mistral-7b

You can now test it on their servers [https://replicate.com/antoinelyset/openhermes-2-mistral-7b](https://replicate.com/antoinelyset/openhermes-2-mistral-7b)
