from transformers import AutoTokenizer
class InstructionPromptSummary():
    
    def __init__(self,model,model_name):
        self.model=model
        self.model_name=model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True) 
        
        


    def create_prompt(self,dataset,example_indices_list,example_index_to_summarize):
        if not example_indices_list:
            dialogue = dataset['test'][example_index_to_summarize]['dialogue']

            prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary:
            """
        else:
            # Initiate an empty prompt. Keep adding more examples for n-shot Inference
            prompt =''
            
            for index in example_indices_list:
                dialogue = dataset['test'][index]['dialogue']
                summary = dataset['test'][index]['summary']
                
                """Create a list of prompts.
                Note: The stop sequence '{summary}\n\n\n' is important for FLAN-T5.
                Other models may have their own preferred stop sequence."""
                
                prompt += f"""
                Dialogue:
                
                {dialogue}
                
                What was going on?
                {summary}
                
                
                """
                
                # Example to summarize
                dialogue = dataset['test'][example_index_to_summarize]['dialogue']
            
            prompt += f"""
        Dialogue:

        {dialogue}

        What was going on?
        """
            
        return prompt



    def n_shot_inference(self,n_shot_prompt,generation_config_params):
        # Tokenizer
        
        inputs = self.tokenizer(n_shot_prompt,return_tensors ='pt')
        
        
        
        output = self.tokenizer.decode(
        self.model.generate(
            inputs["input_ids"],
            generation_config=generation_config_params,
        )[0], 
        skip_special_tokens=True)
        
        return output





        
        
        
    
    
    