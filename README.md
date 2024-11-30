This code repository was created to further research on prompt injection regarding LLM's. Most of them are vulnerable to prompt injection which can confuse the LLM into answering the injected prompt over the user's initial prompt, leading to malicious behavior being executed.

To reproduce the results of this project, you will need to clone the repo.

In order to clone our repo, run this command: git clone https://github.com/Acevortic/CSS_Security.git

NOTE: You will have to create a .env file in the root directory on our own and pass in your own API key for each API you want to use. This project uses:

1. ChatGPT (OpenAI)
2. Gemini (Google)
3. Cohere (Cohere)
4. Claude (Anthropic)

NOTE: You will need to create an API key and import them within the main.py (GPT), GeminiImproved, CohereImproved, and claudeImproved files in order to see any results.

After your API keys have been successfully imported, you can run each file separately, or if running across all the datasets, it is possible to import the files into main.py and call the respective functions.
By default, each file processes only the first 10 examples during testing, and does not write to files. This can and will be changed to iterate through entire datasets and write the output to text files (Hence why you need your own API keys).


The foundational paper for this project:
@misc{https://doi.org/10.48550/arxiv.2302.12173,
  doi = {10.48550/ARXIV.2302.12173},
  url = {https://arxiv.org/abs/2302.12173},
  author = {Greshake, Kai and Abdelnabi, Sahar and Mishra, Shailesh and Endres, Christoph and Holz, Thorsten and Fritz, Mario},
  keywords = {Cryptography and Security (cs.CR), Artificial Intelligence (cs.AI), Computation and Language (cs.CL), Computers and Society (cs.CY), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {More than you've asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}


Further research on LLM prompt injection:
https://arxiv.org/abs/2306.05499
Liu, Yi, et al. "Prompt Injection attack against LLM-integrated Applications." arXiv preprint arXiv:2306.05499 (2023).


