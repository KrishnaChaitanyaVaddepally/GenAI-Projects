import os
import utilities
import random
import streamlit as st
import inflect

from transformers import AutoTokenizer
from transformers import GenerationConfig
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
  
def main():
    # Import Dataset
    huggingface_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(huggingface_dataset_name)

    # Import a pretrained model
    model_name='google/flan-t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    st.title("Dialogue Summarization - Prompt Enginnering")
    
    
    # Number of random numbers to generate (n)
    shots =[0,1,2,3,4,5]
    n = st.selectbox("Select a number for n-shot inference:", shots)
    # dataset_indices = list(range(100))
    # example_indices_list = random.sample(dataset_indices, n)
    example_indices_list = [random.randint(1, 500) for _ in range(n)]
    # st.write(f"example_indices_list:{example_indices_list}")
    
    # Genration Configuration parameters
    st.sidebar.title("Set generation configuration parameters")
    max_new_tokens = st.sidebar.slider("Max new tokens", min_value=1, max_value=150, value=100)
    do_sample = st.sidebar.selectbox("do_sample", [True, False])
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0)
    generation_config_params = GenerationConfig(max_new_tokens=max_new_tokens,
                                                do_sample=do_sample, temperature=temperature)
    
    

    # Create Prompt
    example_index_to_summarize = 200
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    baseline_human_summary = dataset['test'][example_index_to_summarize]['summary']
    
    llm = utilities.InstructionPromptSummary(model,model_name)
    
    
    # example_indices_list = random.choice(range(len(dataset)))
    n_shot_prompt = llm.create_prompt(dataset,example_indices_list,example_index_to_summarize)
    
    model_generated_sumary = llm.n_shot_inference(n_shot_prompt,generation_config_params)

    # Display the generated random numbers
    p = inflect.engine()
    word = p.number_to_words(n)
    # st.markdown(f"{word} shot Inference")
    
    st.write("Dialogue:")

    st.write(f"{dialogue}")
    
    st.write("Baseline Human Summary:")
    st.write(f"{baseline_human_summary}")
    
    st.write("Model generated Summary:")
    st.write(f"{model_generated_sumary}")

if __name__ == "__main__":
    main()

    
    
    
    
    


