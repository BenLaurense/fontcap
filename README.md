# fontcap

Python project to train convolutional NNs to capitalise fonts. 

### Project structure

#### Scraping:

The `fontcap_scraper` package handles font scraping. Fonts are scraped from Google fonts (although the infrastructure i
s set up for more APIs) and are stored as greyscale .png images. The API sources, image size, and font size are configurable.

There is a CLI command `python -m scrape --config <name of config file>`, and an example config is found at `fontcap_scraper/basic_config.yaml`.

There is also a barebones Streamlit UI for viewing the scraped glyphs in `fontcap_scraper/ui.py`.

#### Training:

Code related to training is found in the `fontcap_model` package. Trial model architectures are found in `fontcap_model/models`.

There are CLI commands to run training for selected models: `python -m cli.train_<unet|cnn> -dr <path to data> -ep <num epochs> 
-bs <training batch size> -lr <learning rate>`. Also has options for saving model parameters and profiling data.
