# fontcap

Project to train convolutional NNs to capitalise fonts, as part of my ongoing series of 'finally put all my old projects on GitHub'.

#### Project plan:

* Scrape fonts from Google fonts and similar APIs;
* Encode as low-res greyscale .png images;
* Train a CNN autoencoder/U-Net/more exotic models to capitalise the fonts;

I also want to avoid manual data cleaning as much as possible, and experiment with model-free data cleaning methodologies.

Looking at example outputs from the trained model (`experiments/model_comparison.ipynb`), three things are apparent:
* The raw data is quite dirty
* The models reach minimum test loss quickly (then begin to slowly overfit)
* Even when the models perform well, the outputs are blurry

Point 2 is mostly a consequence of point 1, the fact that data is limited, and lack of HPO. Micro-optimisations in training procedure would be insufficient to increase performance. 
In order to make more progress, take the following steps:

* Add an antialiasing step to the end of the CNN/U-Net models;
* Retrain the CNN/U-Net on the dirty data;
* Extract the latent representation from the bottleneck CNN layer;
* Perform clustering on the latents, and use this to exclude dirty fonts;
* Retrain the CNN/U-Net on this cleaned data.

### Repo structure

#### Scraping:

The `fontcap_scraper` package handles font scraping. Fonts are scraped from Google fonts (although the infrastructure i
s set up for more APIs) and are stored as greyscale .png images. The API sources, image size, and font size are configurable.

There is a CLI command `python -m scrape --config <name of config file>`, and an example config is found at `fontcap_scraper/basic_config.yaml`.

There is also a barebones Streamlit UI for viewing the scraped glyphs in `fontcap_scraper/ui.py`.

#### Training:

Code related to training is found in the `fontcap_model` package. Trial model architectures are found in `fontcap_model/models`.

There are CLI commands to run training for selected models: `python -m cli.train_<unet|cnn> -dr <path to data> -ep <num epochs> 
-bs <training batch size> -lr <learning rate>`. Also has options for saving model parameters and profiling data.

#### Analysis:

Analysis and analysis helpers are found in `experiments` and `analysis`
