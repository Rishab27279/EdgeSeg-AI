# llm_query_simplifier.py - Enhanced LLM for query simplification with comprehensive contextual reasoning

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import re
from typing import List, Dict, Tuple

class QuerySimplifierLLM:
    """LLM for converting complex queries to simple detection prompts using advanced prompting with contextual reasoning"""
    
    def __init__(self, model_name="Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def download_and_setup(self):
        """Download and setup the LLM"""
        print(f"🔄 Downloading {self.model_name}...")
        
        # Download tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Download model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ {self.model_name} downloaded successfully!")
        return True
    
    def create_training_data(self):
        """Placeholder method to maintain compatibility - not used in prompt-based approach"""
        return None
    
    def generate_training_examples(self) -> List[Dict]:
        """Generate comprehensive examples for few-shot prompting including contextual reasoning"""
        return [
            # Preservation examples
            {"complex": "Find a hat", "simple": "Find a hat"},
            {"complex": "Detect sunglasses", "simple": "Detect sunglasses"},
            {"complex": "Locate the speedometer", "simple": "Locate the speedometer"},
            
            # Professional Context Examples
            {"complex": "Show me the safety equipment that construction workers typically wear on their heads", "simple": "Find safety helmet"},
            {"complex": "What protective gear do people use when riding motorcycles?", "simple": "Detect motorcycle helmet"},
            {"complex": "Identify the tool that mechanics use to tighten bolts", "simple": "Find wrench"},
            
            # Activity-Based Context Examples
            {"complex": "What is the person using to communicate in this office setting?", "simple": "Find phone"},
            {"complex": "Show me the equipment used for displaying information to an audience", "simple": "Detect projector"},
            {"complex": "What do people use to stay dry during rainy weather?", "simple": "Locate umbrella"},
            
            # Functional Description Examples
            {"complex": "The device used for measuring vehicle speed in the dashboard", "simple": "Find speedometer"},
            {"complex": "The circular control mechanism used for steering vehicles", "simple": "Locate steering wheel"},
            {"complex": "The instrument used for measuring body temperature in clinical settings", "simple": "Find thermometer"},
            
            # Appearance-Based Descriptions
            {"complex": "The rectangular electronic device with a screen used for computing", "simple": "Find laptop"},
            {"complex": "The transparent protective eyewear worn in bright sunlight", "simple": "Detect sunglasses"},
            {"complex": "The portable communication device that fits in your pocket", "simple": "Locate mobile phone"},
            
            # Medical/Healthcare Technical
            {"complex": "The device that monitors cardiovascular pressure readings", "simple": "Find blood pressure monitor"},
            {"complex": "The electronic instrument for cardiac rhythm monitoring", "simple": "Detect heart monitor"},
            
            # Automotive Technical
            {"complex": "The instrument cluster component that displays engine temperature", "simple": "Find temperature gauge"},
            {"complex": "The safety restraint system used by vehicle occupants", "simple": "Detect seatbelt"},
            
            # Natural Language Requests
            {"complex": "I'm looking for something to protect my eyes from the sun", "simple": "Find sunglasses"},
            {"complex": "Can you help me find the thing I use to make phone calls?", "simple": "Locate mobile phone"},
            {"complex": "Show me what keeps my beverages cold in the kitchen", "simple": "Find refrigerator"},
            
            # Indirect References
            {"complex": "Show me what the chef is using to prepare the meal", "simple": "Find cooking utensil"},
            {"complex": "Point out the item that people sit on during meetings", "simple": "Detect chair"},
            
            # Multi-Step Reasoning / Logical Inference
            {"complex": "In a construction site, what would workers wear to protect their heads from falling objects?", "simple": "Find hard hat"},
            {"complex": "During winter sports, what do people wear to protect their eyes from snow glare?", "simple": "Locate snow goggles"},
            {"complex": "What do people use to stay connected to the internet while traveling?", "simple": "Find laptop"},
            
            # Question Format Conversion
            {"complex": "What do people use to measure time in the kitchen while cooking?", "simple": "Find kitchen timer"},
            {"complex": "How do people protect their hands while handling hot objects?", "simple": "Detect oven mitts"},
            {"complex": "What device helps people navigate while driving?", "simple": "Locate GPS device"}
        ]
    
    def create_dynamic_prompt(self, query: str) -> str:
        """Create context-aware prompt with comprehensive examples and contextual reasoning"""
        
        # Get few-shot examples
        examples = self.generate_training_examples()
        
        # Build the comprehensive prompt with contextual reasoning
        prompt = """You are an advanced query simplification assistant with contextual reasoning capabilities. Convert complex image search queries into simple, natural detection commands.

CORE PRINCIPLES:
- If the query is already simple and clear, keep it exactly as-is
- Use contextual reasoning to understand the intent behind complex descriptions
- Extract the core object from functional, professional, or activity-based contexts
- Preserve natural language flow without robotic templates
- Use varied action words: Find, Locate, Detect, Identify, Spot

TRANSFORMATION PATTERNS:

1. PRESERVE SIMPLE QUERIES (keep unchanged):
   "Find hat" → "Find hat"
   "Detect sunglasses" → "Detect sunglasses"

2. PROFESSIONAL CONTEXT REASONING:
   "Safety equipment construction workers wear on heads" → "Find safety helmet"
   "Tool mechanics use to tighten bolts" → "Find wrench"

3. ACTIVITY-BASED CONTEXT:
   "Equipment for displaying information to audience" → "Detect projector"
   "What people use to stay dry in rain" → "Locate umbrella"

4. FUNCTIONAL DESCRIPTIONS:
   "Device for measuring vehicle speed" → "Find speedometer"
   "Circular control mechanism for steering" → "Locate steering wheel"

5. APPEARANCE-BASED DESCRIPTIONS:
   "Rectangular electronic device with screen for computing" → "Find laptop"
   "Transparent protective eyewear for bright sunlight" → "Detect sunglasses"

6. TECHNICAL TO SIMPLE CONVERSION:
   "Device that monitors cardiovascular pressure readings" → "Find blood pressure monitor"
   "Instrument cluster component displaying engine temperature" → "Find temperature gauge"

7. CONVERSATIONAL TO DIRECT:
   "I'm looking for something to protect my eyes from sun" → "Find sunglasses"
   "Help me find the thing I use to make phone calls" → "Locate mobile phone"

8. INDIRECT REFERENCES:
   "What the chef is using to prepare the meal" → "Find cooking utensil"
   "Item that people sit on during meetings" → "Detect chair"

9. MULTI-STEP REASONING:
   "What would construction workers wear to protect heads from falling objects?" → "Find hard hat"
   "What do people use to stay connected to internet while traveling?" → "Find laptop"

10. QUESTION FORMAT CONVERSION:
    "What do people use to measure time in kitchen while cooking?" → "Find kitchen timer"
    "How do people protect hands while handling hot objects?" → "Detect oven mitts"

COMPLEXITY ASSESSMENT:
- Simple: Already clear → Preserve unchanged
- Moderate: Some description → Natural simplification  
- Complex: Multiple descriptors → Extract core object with context
- Very Complex: Technical/verbose → Focus on main item using reasoning

EXAMPLES:

PRESERVATION (Simple & Clear):
Input: "Find a hat"
Output: "Find a hat"

Input: "Detect sunglasses" 
Output: "Detect sunglasses"

PROFESSIONAL CONTEXT:
Input: "Show me the safety equipment that construction workers typically wear on their heads"
Output: "Find safety helmet"

Input: "What protective gear do people use when riding motorcycles?"
Output: "Detect motorcycle helmet"

ACTIVITY-BASED CONTEXT:
Input: "What is the person using to communicate in this office setting?"
Output: "Find phone"

Input: "Show me the equipment used for displaying information to an audience"
Output: "Detect projector"

FUNCTIONAL DESCRIPTIONS:
Input: "The device used for measuring vehicle speed in the dashboard"
Output: "Find speedometer"

Input: "The circular control mechanism used for steering vehicles"
Output: "Locate steering wheel"

APPEARANCE-BASED:
Input: "The rectangular electronic device with a screen used for computing"
Output: "Find laptop"

Input: "The transparent protective eyewear worn in bright sunlight"
Output: "Detect sunglasses"

TECHNICAL CONVERSION:
Input: "The device that monitors cardiovascular pressure readings"
Output: "Find blood pressure monitor"

Input: "The instrument cluster component that displays engine temperature"
Output: "Find temperature gauge"

CONVERSATIONAL:
Input: "I'm looking for something to protect my eyes from the sun"
Output: "Find sunglasses"

Input: "Show me what keeps my beverages cold in the kitchen"
Output: "Find refrigerator"

MULTI-STEP REASONING:
Input: "In a construction site, what would workers wear to protect their heads from falling objects?"
Output: "Find hard hat"

Input: "What do people use to stay connected to the internet while traveling?"
Output: "Find laptop"

QUESTION FORMAT:
Input: "What do people use to measure time in the kitchen while cooking?"
Output: "Find kitchen timer"

Input: "What device helps people navigate while driving?"
Output: "Locate GPS device"

Now apply contextual reasoning to simplify this query naturally:
Input: "{query}"
Output:"""

        return prompt.format(query=query)
    
    def assess_query_complexity(self, query: str) -> str:
        """Assess the complexity level of the input query with enhanced contextual understanding"""
        query_lower = query.lower().strip()
        word_count = len(query.split())
        
        # Simple patterns
        simple_patterns = [
            r'^(find|detect|locate|identify|spot)\s+(a\s+|an\s+|the\s+)?[\w\s]{1,15}$',
            r'^[\w\s]{1,20}$'
        ]
        
        # Check if it's already simple
        for pattern in simple_patterns:
            if re.match(pattern, query_lower) and word_count <= 4:
                return "simple"
        
        # Check for contextual complexity indicators
        contextual_indicators = [
            'what do people use', 'show me the', 'help me find', 'identify the tool',
            'equipment used for', 'device that monitors', 'instrument for',
            'what would workers', 'how do people', 'in a construction site',
            'during winter sports', 'in this office setting'
        ]
        
        if any(indicator in query_lower for indicator in contextual_indicators):
            return "contextual_complex"
        
        # Check complexity based on descriptors and length
        if word_count <= 6:
            return "moderate"
        elif word_count <= 12:
            return "complex"
        else:
            return "very_complex"
    
    def is_direct_query(self, query: str) -> bool:
        """Enhanced check for direct, simple queries with better pattern recognition"""
        query_lower = query.lower().strip()
        word_count = len(query.split())
        
        # Enhanced patterns for simple queries
        simple_patterns = [
            r'^find\s+(a\s+|an\s+|the\s+)?[\w\s]{1,20}$',
            r'^detect\s+(a\s+|an\s+|the\s+)?[\w\s]{1,20}$',
            r'^locate\s+(a\s+|an\s+|the\s+)?[\w\s]{1,20}$',
            r'^identify\s+(a\s+|an\s+|the\s+)?[\w\s]{1,20}$',
            r'^spot\s+(a\s+|an\s+|the\s+)?[\w\s]{1,20}$'
        ]
        
        # Check if it matches simple patterns and is short
        for pattern in simple_patterns:
            if re.match(pattern, query_lower) and word_count <= 5:
                return True
        
        # Additional check for very simple object names
        if word_count <= 3 and not any(word in query_lower for word in ['what', 'how', 'show', 'help', 'identify the']):
            return True
                
        return False
    
    def normalize_direct_query(self, query: str) -> str:
        """Minimal normalization for direct queries to preserve naturalness"""
        query_clean = query.strip()
        
        # Only fix obvious grammatical issues
        if query_clean.lower().startswith('find sunglass '):
            query_clean = query_clean.replace('sunglass', 'sunglasses')
        
        # Preserve the original query with minimal changes
        return query_clean
    
    def fine_tune_model(self, output_dir="./query_simplifier_model"):
        """Placeholder method to maintain compatibility - not used in prompt-based approach"""
        print("⚠️ Fine-tuning not used in prompt-based approach")
        print("✅ Using advanced prompting strategy with contextual reasoning instead")
        return True
    
    def simplify_query(self, complex_query: str) -> str:
        """Convert complex query to simple detection prompt using advanced prompting with contextual reasoning"""
        
        # First check if it's already a simple direct query
        if self.is_direct_query(complex_query):
            return self.normalize_direct_query(complex_query)
        
        # Use LLM with advanced contextual prompting
        if self.model and self.tokenizer:
            return self.llm_simplify(complex_query)
        else:
            # Fallback to enhanced rule-based with contextual understanding
            return self.rule_based_simplify(complex_query)
    
    def llm_simplify(self, complex_query: str) -> str:
        """Use LLM with advanced contextual prompting to simplify complex query"""
        try:
            # Create dynamic prompt with contextual reasoning
            prompt = self.create_dynamic_prompt(complex_query)
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            # Generate with optimized parameters for contextual reasoning
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 25,  # Slightly longer for contextual responses
                    num_return_sequences=1,
                    temperature=0.1,  # Very low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the output after "Output:"
            if "Output:" in response:
                simplified = response.split("Output:")[-1].strip()
                # Clean up any extra text
                simplified = simplified.split('\n')[0].strip()
                simplified = simplified.split('.')[0].strip()  # Remove trailing periods
                simplified = simplified.split(',')[0].strip()  # Remove trailing commas
                
                if simplified and len(simplified) > 2 and len(simplified) < 100:
                    return simplified
            
            # Fallback to rule-based if LLM output is invalid
            return self.rule_based_simplify(complex_query)
            
        except Exception as e:
            print(f"❌ LLM simplification error: {e}")
            return self.rule_based_simplify(complex_query)
    
    def rule_based_simplify(self, user_query: str) -> str:
        """Enhanced rule-based simplification with contextual understanding and natural language variation"""
        query_lower = user_query.lower().strip()
        
        # Use varied action words for naturalness
        action_words = ["Find", "Locate", "Detect", "Spot"]
        import random
        action = random.choice(action_words)
        
        # Enhanced contextual patterns
        
        # Professional/Occupational Context
        if any(phrase in query_lower for phrase in ['construction workers', 'workers wear', 'safety equipment']):
            if any(word in query_lower for word in ['head', 'heads']):
                return f"{action} safety helmet"
            elif any(word in query_lower for word in ['hand', 'hands']):
                return f"{action} safety gloves"
            else:
                return f"{action} safety equipment"
        
        if any(phrase in query_lower for phrase in ['motorcycle', 'riding', 'protective gear']):
            return f"{action} motorcycle helmet"
        
        if any(phrase in query_lower for phrase in ['mechanics use', 'tighten bolts', 'tool']):
            return f"{action} wrench"
        
        # Activity-Based Context
        if any(phrase in query_lower for phrase in ['office setting', 'communicate', 'phone calls']):
            return f"{action} phone"
        
        if any(phrase in query_lower for phrase in ['displaying information', 'audience', 'presentation']):
            return f"{action} projector"
        
        if any(phrase in query_lower for phrase in ['rainy weather', 'stay dry', 'rain']):
            return f"{action} umbrella"
        
        # Functional Descriptions
        if any(phrase in query_lower for phrase in ['measuring vehicle speed', 'speed measurement', 'dashboard']):
            return f"{action} speedometer"
        
        if any(phrase in query_lower for phrase in ['steering vehicles', 'circular control', 'steering']):
            return f"{action} steering wheel"
        
        if any(phrase in query_lower for phrase in ['body temperature', 'clinical settings', 'thermometer']):
            return f"{action} thermometer"
        
        # Appearance-Based Descriptions
        if any(phrase in query_lower for phrase in ['rectangular electronic device', 'screen used for computing', 'laptop']):
            return f"{action} laptop"
        
        if any(phrase in query_lower for phrase in ['transparent protective eyewear', 'bright sunlight', 'protect eyes from sun']):
            return f"{action} sunglasses"
        
        if any(phrase in query_lower for phrase in ['portable communication device', 'fits in pocket', 'mobile']):
            return f"{action} mobile phone"
        
        # Medical/Healthcare Technical
        if any(phrase in query_lower for phrase in ['cardiovascular pressure', 'blood pressure', 'pressure readings']):
            return f"{action} blood pressure monitor"
        
        if any(phrase in query_lower for phrase in ['cardiac rhythm', 'heart monitor', 'heart rhythm']):
            return f"{action} heart monitor"
        
        # Automotive Technical
        if any(phrase in query_lower for phrase in ['engine temperature', 'temperature gauge', 'instrument cluster']):
            return f"{action} temperature gauge"
        
        if any(phrase in query_lower for phrase in ['safety restraint', 'vehicle occupants', 'seatbelt']):
            return f"{action} seatbelt"
        
        # Kitchen/Cooking Context
        if any(phrase in query_lower for phrase in ['keeps beverages cold', 'kitchen cold', 'refrigerator']):
            return f"{action} refrigerator"
        
        if any(phrase in query_lower for phrase in ['chef using', 'prepare meal', 'cooking utensil']):
            return f"{action} cooking utensil"
        
        if any(phrase in query_lower for phrase in ['measure time', 'kitchen timer', 'cooking timer']):
            return f"{action} kitchen timer"
        
        if any(phrase in query_lower for phrase in ['protect hands', 'hot objects', 'oven mitts']):
            return f"{action} oven mitts"
        
        # Multi-Step Reasoning
        if any(phrase in query_lower for phrase in ['construction site', 'falling objects', 'protect heads']):
            return f"{action} hard hat"
        
        if any(phrase in query_lower for phrase in ['winter sports', 'snow glare', 'protect eyes']):
            return f"{action} snow goggles"
        
        if any(phrase in query_lower for phrase in ['connected to internet', 'while traveling', 'internet traveling']):
            return f"{action} laptop"
        
        if any(phrase in query_lower for phrase in ['navigate while driving', 'navigation', 'gps']):
            return f"{action} GPS device"
        
        # Meeting/Office Context
        if any(phrase in query_lower for phrase in ['people sit on', 'during meetings', 'meeting chair']):
            return f"{action} chair"
        
        # Original keyword-based patterns (enhanced)
        # Sun protection / Sunglasses
        if any(word in query_lower for word in ['sunglass', 'sunglasses', 'sun protection', 'uv protection', 'shades', 'dark glasses', 'tinted eyewear']):
            return f"{action} sunglasses"
        
        # Automotive keywords
        elif any(word in query_lower for word in ['speed', 'speedometer', 'mph', 'kmh']):
            return f"{action} speedometer"
        elif any(word in query_lower for word in ['dashboard', 'instrument panel', 'controls']):
            return f"{action} car dashboard"
        elif any(word in query_lower for word in ['steering', 'wheel', 'driving']):
            return f"{action} steering wheel"
        elif any(word in query_lower for word in ['gauge', 'dial', 'meter', 'instrument']):
            return f"{action} gauges"
        
        # Safety keywords
        elif any(word in query_lower for word in ['safety', 'protective', 'protection', 'ppe']):
            return f"{action} safety equipment"
        elif any(word in query_lower for word in ['helmet', 'hard hat', 'head protection']):
            return f"{action} safety helmet"
        elif any(word in query_lower for word in ['gloves', 'hand protection']):
            return f"{action} safety gloves"
        elif any(word in query_lower for word in ['vest', 'high visibility', 'hi-vis']):
            return f"{action} safety vest"
        
        # Electronics keywords
        elif any(word in query_lower for word in ['electronic', 'device', 'gadget', 'technology']):
            return f"{action} electronic devices"
        elif any(word in query_lower for word in ['phone', 'mobile', 'smartphone']):
            return f"{action} mobile phone"
        elif any(word in query_lower for word in ['laptop', 'computer', 'notebook']):
            return f"{action} laptop"
        elif any(word in query_lower for word in ['monitor', 'screen', 'display']):
            return f"{action} monitor"
        
        # Medical keywords
        elif any(word in query_lower for word in ['medical', 'healthcare', 'hospital']):
            return f"{action} medical equipment"
        elif any(word in query_lower for word in ['thermometer', 'temperature']):
            return f"{action} thermometer"
        elif any(word in query_lower for word in ['blood pressure', 'bp monitor']):
            return f"{action} blood pressure monitor"
        
        # Kitchen keywords
        elif any(word in query_lower for word in ['kitchen', 'cooking', 'appliance']):
            return f"{action} kitchen appliances"
        elif any(word in query_lower for word in ['stove', 'cooktop', 'burner']):
            return f"{action} stove"
        elif any(word in query_lower for word in ['refrigerator', 'fridge']):
            return f"{action} refrigerator"
        
        # Tools keywords
        elif any(word in query_lower for word in ['tool', 'hammer', 'screwdriver']):
            return f"{action} tools"
        elif any(word in query_lower for word in ['drill', 'drilling']):
            return f"{action} drill"
        elif any(word in query_lower for word in ['wrench', 'tighten', 'bolts']):
            return f"{action} wrench"
        
        # Sports keywords
        elif any(word in query_lower for word in ['sports', 'ball', 'game']):
            return f"{action} sports equipment"
        elif any(word in query_lower for word in ['bicycle', 'bike']):
            return f"{action} bicycle"
        
        # Furniture keywords
        elif any(word in query_lower for word in ['furniture', 'chair', 'table']):
            return f"{action} furniture"
        elif any(word in query_lower for word in ['bed', 'mattress']):
            return f"{action} bed"
        
        # Weather keywords
        elif any(word in query_lower for word in ['weather', 'rain', 'umbrella']):
            return f"{action} umbrella"
        elif any(word in query_lower for word in ['hat', 'cap']):
            return f"{action} hat"
        
        # Extract object name if simple pattern
        simple_patterns = [
            r'find (\w+)',
            r'detect (\w+)',
            r'locate (\w+)',
            r'identify (\w+)'
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, query_lower)
            if match:
                object_name = match.group(1)
                return f"Find {object_name}"
        
        # Default fallback with natural variation
        return f"{action} objects"

# Usage example
def main():
    """Main function to test the enhanced prompt-based query simplifier with contextual reasoning"""
    
    # Initialize the query simplifier
    simplifier = QuerySimplifierLLM()
    
    # Download and setup the model
    if simplifier.download_and_setup():
        print("✅ Model downloaded successfully!")
        
        # Test the model with various queries including contextual reasoning examples
        test_queries = [
            # Simple preservation
            "Find a hat",
            "Detect sunglasses",
            
            # Professional context
            "Show me the safety equipment that construction workers typically wear on their heads",
            "What protective gear do people use when riding motorcycles?",
            "Identify the tool that mechanics use to tighten bolts",
            
            # Activity-based context
            "What is the person using to communicate in this office setting?",
            "Show me the equipment used for displaying information to an audience",
            "What do people use to stay dry during rainy weather?",
            
            # Functional descriptions
            "The device used for measuring vehicle speed in the dashboard",
            "The circular control mechanism used for steering vehicles",
            "The instrument used for measuring body temperature in clinical settings",
            
            # Appearance-based descriptions
            "The rectangular electronic device with a screen used for computing",
            "The transparent protective eyewear worn in bright sunlight",
            "The portable communication device that fits in your pocket",
            
            # Technical conversions
            "The device that monitors cardiovascular pressure readings",
            "The instrument cluster component that displays engine temperature",
            
            # Conversational
            "I'm looking for something to protect my eyes from the sun",
            "Can you help me find the thing I use to make phone calls?",
            "Show me what keeps my beverages cold in the kitchen",
            
            # Multi-step reasoning
            "In a construction site, what would workers wear to protect their heads from falling objects?",
            "During winter sports, what do people wear to protect their eyes from snow glare?",
            "What do people use to stay connected to the internet while traveling?",
            
            # Question format
            "What do people use to measure time in the kitchen while cooking?",
            "How do people protect their hands while handling hot objects?",
            "What device helps people navigate while driving?"
        ]
        
        print("\n🧪 Testing the enhanced contextual reasoning model:")
        for query in test_queries:
            simplified = simplifier.simplify_query(query)
            print(f"Complex: {query}")
            print(f"Simple: {simplified}")
            print("-" * 70)

if __name__ == "__main__":
    main()
