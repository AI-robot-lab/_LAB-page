# Vision-Language Models (VLM)

## Wprowadzenie

**Vision-Language Models** to zaawansowane modele AI łączące rozumienie obrazów i języka naturalnego. W robotyce humanoidalnej VLM umożliwiają robotom "widzenie ze zrozumieniem" - interpretację scen wizualnych i naturalną komunikację o tym, co widzą.

## Popularne Modele VLM

### CLIP (Contrastive Language-Image Pre-training)

```python
import torch
import clip
from PIL import Image

# Załaduj model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Przygotuj obraz i tekst
image = preprocess(Image.open("robot_scene.jpg")).unsqueeze(0).to(device)
text = clip.tokenize([
    "robot picking up a cup",
    "empty table",
    "person walking",
    "door is open"
]).to(device)

# Inference
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Similarity
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Prawdopodobieństwa:")
for desc, prob in zip(text_descriptions, probs[0]):
    print(f"{desc}: {prob*100:.2f}%")
```

### BLIP (Bootstrapping Language-Image Pre-training)

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

# Image captioning
image = Image.open("robot_workspace.jpg")

# Unconditional generation
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")

# Conditional generation (z promptem)
text = "a robot is"
inputs = processor(image, text, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Conditional: {caption}")
```

### LLaVA (Large Language and Vision Assistant)

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates

# Load model
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

def ask_about_image(image_path, question):
    """
    Zadaj pytanie o obraz
    """
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    
    # Prepare prompt
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.cuda(),
            max_new_tokens=512,
            use_cache=True
        )
    
    response = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return response

# Użycie
answer = ask_about_image(
    "robot_scene.jpg",
    "What objects can the robot grasp in this scene?"
)
print(answer)
```

## VLM dla Robotyki

### Object Grounding

```python
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class VLMObjectDetector:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
    
    def detect_objects(self, image, text_queries):
        """
        Zero-shot object detection z opisami tekstowymi
        
        Args:
            image: PIL Image
            text_queries: Lista opisów (np. ["red cup", "laptop"])
        
        Returns:
            Detekcje z bounding boxes
        """
        inputs = self.processor(
            text=text_queries, 
            images=image, 
            return_tensors="pt"
        )
        
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )
        
        detections = []
        for i, query in enumerate(text_queries):
            boxes = results[0]['boxes'][results[0]['labels'] == i]
            scores = results[0]['scores'][results[0]['labels'] == i]
            
            for box, score in zip(boxes, scores):
                if score > 0.3:
                    detections.append({
                        'object': query,
                        'bbox': box.tolist(),
                        'confidence': score.item()
                    })
        
        return detections

# Użycie
detector = VLMObjectDetector()
image = Image.open("workspace.jpg")

detections = detector.detect_objects(
    image,
    ["cup", "phone", "keyboard", "mouse"]
)

for det in detections:
    print(f"{det['object']}: {det['confidence']:.2f} at {det['bbox']}")
```

### Visual Question Answering (VQA)

```python
class RobotVQA:
    def __init__(self):
        from transformers import ViltProcessor, ViltForQuestionAnswering
        
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
    
    def answer_question(self, image, question):
        """
        Odpowiedz na pytanie o obraz
        """
        # Prepare inputs
        encoding = self.processor(image, question, return_tensors="pt")
        
        # Forward pass
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        # Decode answer
        answer = self.model.config.id2label[idx]
        
        return answer

# Użycie w robotyce
vqa = RobotVQA()
image = robot_camera.capture()

# Pytania pomocnicze dla decyzji robota
questions = [
    "Is the cup empty?",
    "What color is the object?",
    "Is there a person in the scene?",
    "Is the door open?"
]

for q in questions:
    answer = vqa.answer_question(image, q)
    print(f"Q: {q}\nA: {answer}\n")
```

## Multimodal Chain-of-Thought

```python
class MultimodalReasoning:
    def __init__(self, vlm_model, llm_model):
        self.vlm = vlm_model  # np. LLaVA
        self.llm = llm_model  # np. Claude, GPT-4
    
    def reason_about_scene(self, image, task):
        """
        Multi-step reasoning o scenie wizualnej
        """
        # Krok 1: Opisz scenę
        scene_description = self.vlm.caption(image)
        
        # Krok 2: Zidentyfikuj obiekty
        objects = self.vlm.detect_objects(
            image, 
            ["cup", "plate", "bottle", "utensils"]
        )
        
        # Krok 3: Reasoning z LLM
        prompt = f"""
Scena: {scene_description}

Obiekty wykryte:
{[obj['object'] for obj in objects]}

Zadanie: {task}

Przemyśl krok po kroku:
1. Jakie obiekty są potrzebne do zadania?
2. Gdzie one są?
3. Jaka jest optymalna kolejność akcji?
4. Czy są jakieś przeszkody?

Plan działania:
"""
        
        plan = self.llm.generate(prompt)
        
        return {
            'scene': scene_description,
            'objects': objects,
            'plan': plan
        }

# Użycie
reasoning = MultimodalReasoning(llava_model, claude_api)
result = reasoning.reason_about_scene(
    image=camera.capture(),
    task="Prepare the table for dinner"
)

print(result['plan'])
```

## Image Segmentation z Language

```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class LanguageGuidedSegmentation:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
    
    def segment_by_text(self, image, prompts):
        """
        Segmentacja obrazu według opisów tekstowych
        
        Args:
            image: PIL Image
            prompts: Lista opisów (np. ["the red cup", "the table"])
        """
        inputs = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt"
        )
        
        outputs = self.model(**inputs)
        
        # Get masks
        masks = outputs.logits
        
        results = []
        for i, prompt in enumerate(prompts):
            mask = torch.sigmoid(masks[i])
            results.append({
                'prompt': prompt,
                'mask': mask.squeeze().cpu().numpy()
            })
        
        return results

# Użycie
segmenter = LanguageGuidedSegmentation()
image = Image.open("table.jpg")

segments = segmenter.segment_by_text(
    image,
    ["the cup on the left", "the plate", "the table surface"]
)

# Wizualizacja
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(segments), figsize=(15, 5))
for ax, seg in zip(axes, segments):
    ax.imshow(seg['mask'], cmap='hot')
    ax.set_title(seg['prompt'])
    ax.axis('off')
plt.show()
```

## Aplikacje w Robotyce

### 1. Scene Understanding dla Manipulacji

```python
class SceneUnderstandingPipeline:
    def __init__(self):
        self.vlm = LLaVAModel()
        self.grounding = OwlViTDetector()
        self.segmentation = CLIPSegModel()
    
    def analyze_manipulation_scene(self, image, target_object):
        """
        Pełna analiza sceny dla zadania manipulacji
        """
        # 1. Global understanding
        scene_desc = self.vlm.caption(image)
        
        # 2. Locate target
        detections = self.grounding.detect([target_object], image)
        
        if not detections:
            return None
        
        target_bbox = detections[0]['bbox']
        
        # 3. Check reachability
        reachability = self.vlm.ask(
            image,
            f"Is the {target_object} reachable without moving other objects?"
        )
        
        # 4. Segment for precise grasp
        mask = self.segmentation.segment(image, target_object)
        
        # 5. Grasp point estimation
        grasp_question = f"Where should a robot gripper grasp the {target_object}?"
        grasp_guidance = self.vlm.ask(image, grasp_question)
        
        return {
            'scene': scene_desc,
            'target_location': target_bbox,
            'reachable': reachability,
            'segmentation_mask': mask,
            'grasp_guidance': grasp_guidance
        }
```

### 2. Human-Robot Interaction

```python
class VLMInteraction:
    def __init__(self):
        self.vlm = LLaVAModel()
    
    def understand_gesture(self, image):
        """
        Rozpoznawanie gestów i intencji człowieka
        """
        questions = [
            "What is the person pointing at?",
            "What gesture is the person making?",
            "What might the person want the robot to do?"
        ]
        
        responses = {}
        for q in questions:
            responses[q] = self.vlm.ask(image, q)
        
        return responses
    
    def explain_action(self, image, action):
        """
        Robot wyjaśnia swoje działanie
        """
        prompt = f"""
Look at this scene. The robot is about to {action}.
Explain to a human why this is the appropriate action.
"""
        explanation = self.vlm.generate(image, prompt)
        return explanation
```

## Fine-tuning VLM dla Robotyki

```python
from transformers import Trainer, TrainingArguments

class RobotVLMFineTuner:
    def __init__(self, base_model="Salesforce/blip-image-captioning-base"):
        from transformers import BlipForConditionalGeneration
        
        self.model = BlipForConditionalGeneration.from_pretrained(base_model)
        self.processor = BlipProcessor.from_pretrained(base_model)
    
    def prepare_dataset(self, robot_data):
        """
        Przygotuj dataset z danych robotycznych
        
        robot_data format:
        [
            {'image': PIL.Image, 'caption': 'robot picking up red cube'},
            ...
        ]
        """
        processed = []
        
        for item in robot_data:
            inputs = self.processor(
                images=item['image'],
                text=item['caption'],
                return_tensors="pt",
                padding=True
            )
            processed.append(inputs)
        
        return processed
    
    def train(self, train_dataset, val_dataset):
        """
        Fine-tune na danych robotycznych
        """
        training_args = TrainingArguments(
            output_dir="./robot-vlm",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            warmup_steps=500,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        
        return self.model
```

## Porównanie Modeli VLM

| Model | Parametry | Mocne Strony | Zastosowanie |
|-------|-----------|--------------|--------------|
| **CLIP** | 400M | Zero-shot classification | Object recognition |
| **BLIP** | 385M | Image captioning | Scene description |
| **LLaVA** | 7B-13B | Visual reasoning | Complex Q&A |
| **Owl-ViT** | 142M | Zero-shot detection | Object grounding |
| **CLIPSeg** | 85M | Language-guided segmentation | Precise localization |

## Best Practices

### 1. Preprocessing

```python
def preprocess_robot_image(image):
    """
    Przygotuj obraz z kamery robota dla VLM
    """
    # Resize do standardowego rozmiaru
    image = image.resize((224, 224))
    
    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)
```

### 2. Prompt Engineering

```python
def create_robot_prompt(task, scene_info):
    """
    Twórz efektywne prompty dla VLM
    """
    prompt = f"""
You are a robot vision system. Analyze this scene carefully.

Task: {task}

Context:
- Environment: {scene_info['environment']}
- Available objects: {scene_info['objects']}
- Constraints: {scene_info['constraints']}

Provide:
1. What you see
2. Relevant objects for the task
3. Recommended action sequence
"""
    return prompt
```

## Powiązane Artykuły

- [LLM](#wiki-llm) - Large Language Models
- [Computer Vision](#wiki-computer-vision)
- [Deep Learning](#wiki-deep-learning)
- [Transformers](#wiki-transformers)

## Zasoby

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
