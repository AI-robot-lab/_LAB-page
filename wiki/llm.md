# Large Language Models (LLM)

## Wprowadzenie

**Large Language Models** to modele głębokiego uczenia trenowane na ogromnych korpusach tekstowych, zdolne do rozumienia i generowania języka naturalnego. W robotyce humanoidalnej LLM-y umożliwiają naturalną komunikację, reasoning i planowanie wysokopoziomowe.

## Architektura Transformer

Podstawa nowoczesnych LLM:

```
Input → Embedding → Positional Encoding
        ↓
Multi-Head Attention → Add & Norm
        ↓
Feed Forward → Add & Norm
        ↓
[Repeat N layers]
        ↓
Output Logits → Softmax
```

### Self-Attention Mechanism

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        """
        # (batch, heads, seq_len, d_k) @ (batch, heads, d_k, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        output, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear
        output = self.W_o(output)
        
        return output, attention
```

## Popularne Modele

### GPT (Generative Pre-trained Transformer)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPTGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        """
        Generate text from prompt
        """
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        return generated_text

# Użycie
generator = GPTGenerator()
response = generator.generate(
    "Robot powinien pomóc użytkownikowi przez"
)
print(response)
```

### Claude (Anthropic)

```python
from anthropic import Anthropic

class ClaudeAssistant:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
    
    def chat(self, user_message):
        """
        Chat with Claude
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset(self):
        self.conversation_history = []

# Użycie w robotyce
assistant = ClaudeAssistant(api_key="your-key")

# Reasoning o akcjach robota
response = assistant.chat("""
Użytkownik prosi: "Podaj mi wodę"
Kontekst: Robot widzi szklankę na stole 2m przed sobą
Zadanie: Zaplanuj sekwencję akcji robota
""")

print(response)
```

## Prompting Techniques

### Few-Shot Learning

```python
def create_few_shot_prompt(task, examples, query):
    """
    Few-shot learning dla LLM
    """
    prompt = f"Zadanie: {task}\n\n"
    
    for i, (input_text, output_text) in enumerate(examples, 1):
        prompt += f"Przykład {i}:\n"
        prompt += f"Input: {input_text}\n"
        prompt += f"Output: {output_text}\n\n"
    
    prompt += f"Teraz Twoja kolej:\n"
    prompt += f"Input: {query}\n"
    prompt += f"Output:"
    
    return prompt

# Użycie
examples = [
    ("Idź do kuchni", "navigate(location='kitchen')"),
    ("Przynieś kubek", "grasp(object='cup'); navigate(location='user')"),
]

prompt = create_few_shot_prompt(
    task="Przetłumacz polecenie na kod robota",
    examples=examples,
    query="Podaj mi wodę"
)
```

### Chain-of-Thought

```python
def chain_of_thought_prompt(problem):
    """
    CoT prompting dla złożonych zadań
    """
    prompt = f"""
Rozwiąż ten problem krok po kroku:

Problem: {problem}

Myślmy o tym krok po kroku:
1. Najpierw zidentyfikujmy
2. Następnie rozważmy
3. W końcu określmy

Odpowiedź:
"""
    return prompt

# Dla robotyki
problem = """
Użytkownik prosi robota o przygotowanie kawy.
Robot widzi: ekspres do kawy, kubek, mleko w lodówce.
Stan baterii: 30%.
Jakie akcje powinien wykonać robot?
"""

response = llm.generate(chain_of_thought_prompt(problem))
```

### ReAct (Reasoning + Acting)

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def run(self, task):
        """
        ReAct loop: Thought → Action → Observation
        """
        context = f"Zadanie: {task}\n\n"
        max_iterations = 5
        
        for i in range(max_iterations):
            # Thought
            thought_prompt = context + "Thought:"
            thought = self.llm.generate(thought_prompt)
            context += f"Thought: {thought}\n"
            
            # Action
            action_prompt = context + "Action:"
            action = self.llm.generate(action_prompt)
            context += f"Action: {action}\n"
            
            # Execute action
            if action.startswith("FINISH"):
                break
            
            observation = self.execute_tool(action)
            context += f"Observation: {observation}\n\n"
        
        return context
    
    def execute_tool(self, action):
        """
        Execute tool based on action string
        """
        # Parse action
        # "navigate(kitchen)" → wywołaj navigate tool
        for tool in self.tools:
            if tool.name in action:
                return tool.execute(action)
        
        return "Tool not found"
```

## Fine-Tuning dla Robotyki

### Supervised Fine-Tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Przygotowanie datasetu
robot_instructions = [
    {
        "input": "Idź do kuchni",
        "output": "```python\nrobot.navigate(location='kitchen')\n```"
    },
    {
        "input": "Podaj mi wodę",
        "output": "```python\nrobot.grasp(object='glass')\nrobot.pour_water()\nrobot.navigate_to_user()\nrobot.handover()\n```"
    },
    # ... więcej przykładów
]

dataset = Dataset.from_list(robot_instructions)

# Tokenization
def tokenize_function(examples):
    full_text = [
        f"Instruction: {inp}\nResponse: {out}" 
        for inp, out in zip(examples['input'], examples['output'])
    ]
    return tokenizer(full_text, truncation=True, padding='max_length')

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tuning
training_args = TrainingArguments(
    output_dir='./robot-llm',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

# LoRA konfiguracja
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Zastosuj LoRA do modelu
model = get_peft_model(base_model, lora_config)

print(f"Trainable parameters: {model.print_trainable_parameters()}")
# Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06
```

## Retrieval-Augmented Generation (RAG)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, llm):
        self.llm = llm
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FAISS index
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.knowledge_base = []
    
    def add_documents(self, documents):
        """
        Dodaj dokumenty do knowledge base
        """
        # Encode
        embeddings = self.embedder.encode(documents)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.knowledge_base.extend(documents)
    
    def retrieve(self, query, k=3):
        """
        Retrieve top-k najbardziej relevantnych dokumentów
        """
        query_embedding = self.embedder.encode([query])
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        retrieved_docs = [self.knowledge_base[i] for i in indices[0]]
        
        return retrieved_docs
    
    def generate(self, query):
        """
        RAG: Retrieve → Augment → Generate
        """
        # Retrieve
        relevant_docs = self.retrieve(query)
        
        # Augment prompt
        context = "\n\n".join(relevant_docs)
        augmented_prompt = f"""
Kontekst:
{context}

Pytanie: {query}

Odpowiedź:"""
        
        # Generate
        response = self.llm.generate(augmented_prompt)
        
        return response

# Użycie
rag = RAGSystem(llm)

# Dodaj dokumentację robota
docs = [
    "Robot Unitree G1 ma 23 stopnie swobody...",
    "Procedura chwytania: 1) Zbliż się do obiektu...",
    "Bateria: Ładowanie trwa 2 godziny, zasięg 4 godziny pracy..."
]
rag.add_documents(docs)

response = rag.generate("Jak długo trwa ładowanie robota?")
```

## LLM dla Task Planning

```python
class RobotTaskPlanner:
    def __init__(self, llm):
        self.llm = llm
        self.available_actions = [
            "navigate(location)",
            "grasp(object)",
            "release()",
            "pour(container, amount)",
            "speak(message)",
            "wait(seconds)"
        ]
    
    def plan(self, task, world_state):
        """
        Generate action plan for task
        """
        prompt = f"""
Jesteś planerem zadań dla robota humanoidalnego.

Dostępne akcje:
{', '.join(self.available_actions)}

Stan świata:
{world_state}

Zadanie użytkownika: {task}

Wygeneruj sekwencję akcji w formacie JSON:
```json
{{
    "plan": [
        {{"action": "nazwa_akcji", "params": {{}}, "reasoning": "dlaczego"}},
        ...
    ]
}}
```

Plan:"""
        
        response = self.llm.generate(prompt)
        
        # Parse JSON
        import json
        plan_data = json.loads(
            response.split("```json")[1].split("```")[0]
        )
        
        return plan_data['plan']
    
    def execute_plan(self, plan):
        """
        Execute planned actions
        """
        for step in plan:
            action = step['action']
            params = step['params']
            
            print(f"Executing: {action} with {params}")
            print(f"Reasoning: {step['reasoning']}")
            
            # Wywołaj rzeczywistą akcję robota
            # self.robot.execute(action, params)
```

## Evaluation Metrics

```python
def calculate_perplexity(model, text, tokenizer):
    """
    Perplexity - miara jakości modelu językowego
    """
    encodings = tokenizer(text, return_tensors='pt')
    
    max_length = model.config.n_positions
    stride = 512
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    return ppl.item()
```

## Integracja z ROS2

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class LLMReasoningNode(Node):
    def __init__(self):
        super().__init__('llm_reasoning')
        
        self.llm = ClaudeAssistant(api_key="your-key")
        
        # Subscriber
        self.command_sub = self.create_subscription(
            String,
            '/user/command',
            self.command_callback,
            10
        )
        
        # Publisher
        self.action_pub = self.create_publisher(
            String,
            '/robot/action',
            10
        )
    
    def command_callback(self, msg):
        user_command = msg.data
        
        # Reasoning z LLM
        response = self.llm.chat(f"""
Użytkownik: {user_command}

Wygeneruj akcję robota w formacie:
action_type: nazwa_akcji
parameters: {{}}
""")
        
        # Publikuj akcję
        action_msg = String()
        action_msg.data = response
        self.action_pub.publish(action_msg)
```

## Powiązane Artykuły

- [VLM](#wiki-vlm) - Vision-Language Models
- [Framework PCA](#wiki-pca-framework) - Kognicja
- [Deep Learning](#wiki-deep-learning)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
