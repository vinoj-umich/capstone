import torch
import pandas as pd

from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForQuestionAnswering

class QuestionAnswerGenerator:
    def __init__(self):
        # Load Doc2Query model for question generation
        self.model_name = 'doc2query/all-with_prefix-t5-base-v1'
        self.qgen_tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.qgen_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Load BERT or similar model for Question Answering (QA)
        self.qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def generate_questions(self, chunk, num_questions=5):
        """
        Generate questions from a chunk of text using the Doc2Query model.

        :param chunk: The input chunk of text to generate questions for.
        :param num_questions: The number of questions to generate (default is 5).
        :return: A list of generated questions.
        """
        # Prepare the chunk for Doc2Query
        input_text = f"generate questions: {chunk}"
        inputs = self.qgen_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)

        # Check if we are using greedy decoding or beam search
        if num_questions == 1:
            # Use greedy decoding for one question
            outputs = self.qgen_model.generate(
                **inputs, 
                max_length=50, 
                num_return_sequences=1,  # Only generate one question
                no_repeat_ngram_size=2
            )
        else:
            # Use beam search for multiple questions
            outputs = self.qgen_model.generate(
                **inputs, 
                max_length=150, 
                num_return_sequences=num_questions,  # Generate multiple questions
                num_beams=num_questions,  # Use beam search
                no_repeat_ngram_size=2
            )

        # Decode the generated questions
        questions = [self.qgen_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return questions

    def generate_answers(self, chunk, questions):
        """
        Generate answers for a list of questions given a chunk of text.

        :param chunk: The input chunk of text for answering the questions.
        :param questions: A list of questions to answer.
        :return: A list of answers corresponding to the input questions.
        """
        answers = []
        for question in questions:
            # Encode the question and the context (chunk) for QA model with truncation and padding
            inputs = self.qa_tokenizer.encode_plus(
                question, 
                chunk, 
                return_tensors='pt', 
                truncation=True,  # Ensure the input sequence is truncated to fit the max length
                padding=True,     # Pad the sequence if it's shorter than the maximum length
                max_length=512    # Set the max length for the sequence
            )

            # Get the start and end positions of the answer
            outputs = self.qa_model(**inputs)

            # If the model outputs a tuple (start_scores, end_scores)
            if isinstance(outputs, tuple):
                answer_start_scores, answer_end_scores = outputs
            else:
                # If the model returns a dict, extract the start and end scores
                answer_start_scores = outputs['start_logits']
                answer_end_scores = outputs['end_logits']

            # Get the most likely beginning and end of the answer
            start_index = torch.argmax(answer_start_scores)
            end_index = torch.argmax(answer_end_scores)

            # Decode the answer from the token indices
            answer = self.qa_tokenizer.convert_tokens_to_string(
                self.qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index+1])
            )

            answers.append(answer)
        return answers

    def transform(self, chunk_data):
        """
        Transform the input chunk data by generating questions and answers.

        :param chunk_data: A list of chunks, each containing a sentence chunk.
        :return: A list of chunks with generated questions and answers added.
        """
        all_chunk_qa = []
        for chunk in chunk_data:
            chunk_text = chunk['sentence_chunk']  # Get the text of the chunk
            questions = self.generate_questions(chunk_text)
            answers = self.generate_answers(chunk_text, questions)
            chunk['generated_questions'] = questions
            chunk['generated_answers'] = answers
            all_chunk_qa.append(chunk)
        return all_chunk_qa