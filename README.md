# teknium/OpenHermes-2-Mistral-7B Cog model

This is an implementation of the [teknium/OpenHermes-2-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2-Mistral-7B) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="Explain in a short paragraph quantu field theory to a high school student"
