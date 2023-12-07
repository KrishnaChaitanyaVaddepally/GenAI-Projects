# Enhanced Dialogue Summarization with FLAN-T5
This project guides you through the process of fine-tuning an existing Language Model (LLM) from Hugging Face for enhanced dialogue summarization. The primary model used for this task is the FLAN-T5, a high-quality instruction-tuned model capable of summarizing text out of the box.


## Getting Started

1. **Clone this repository to your local machine:**

   ```bash
   git clone https://github.com/KrishnaChaitanyaVaddepally/GenAI-Projects/FineTuning.git

2. **Install the required dependencies:**
 bash
   #pip install -r requirements.txt```

3. **Open and run app.py**

# Overview
In this notebook, you will:

Utilize the FLAN-T5 model for dialogue summarization.
Explore a full fine-tuning approach to enhance inferences.
Evaluate the fine-tuned model using ROUGE metrics.
Perform Parameter Efficient Fine-Tuning (PEFT).
Evaluate the resulting model and observe the benefits of PEFT, which outweigh slightly lower performance metrics.
# Steps
1. Preprocess the Dialog-Summary Dataset
  - Fine-Tune the Model with the Preprocessed Dataset
  - Evaluate the Model Qualitatively (Human Evaluation)
  - Evaluate the Model Quantitatively (with ROUGE Metric)
2. Perform Parameter Efficient Fine-Tuning (PEFT)
  - Setup the PEFT/LoRA model for Fine-Tuning
  - Train PEFT Adapter
  - Evaluate the Model Qualitatively (Human Evaluation)
  - Evaluate the Model Quantitatively (with ROUGE Metric)

# Note
Ensure that you have installed the required dependencies, dataset and LLM.

Happy fine-tuning and summarizing! 🚀
