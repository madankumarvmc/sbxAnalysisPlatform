"""
Google Gemini API Client for Word Report Generation

This module handles all interactions with the Gemini API for generating
intelligent insights and analysis text for the Word reports.
"""

import os
import time
from typing import Optional

# Try to import dependencies with graceful fallback
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional
    pass

class GeminiClient:
    """
    Client for interacting with Google's Gemini API.
    
    Features:
    - Automatic retry on failures
    - Rate limiting protection
    - Fallback text support
    - Error logging
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-1.5-flash'):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Optional API key (defaults to environment variable)
            model_name: Gemini model to use (default: gemini-1.5-flash)
        """
        # Check if Gemini is available
        if not GEMINI_AVAILABLE:
            self.available = False
            self.api_key = None
            self.model = None
            return
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            self.available = False
            self.model = None
            return
        
        # Configure the API
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.available = True
        except Exception as e:
            self.available = False
            self.model = None
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.timeout = 30  # seconds
    
    def generate_insight(self, prompt: str, fallback_text: str = "", temperature: float = 0.7) -> str:
        """
        Generate insight text using Gemini API.
        
        Args:
            prompt: The prompt to send to Gemini
            fallback_text: Text to use if API call fails
            temperature: Creativity parameter (0.0 to 1.0)
        
        Returns:
            Generated text or fallback text if generation fails
        """
        # Return fallback if not available
        if not getattr(self, 'available', False) or not self.model:
            return fallback_text
        
        if not prompt:
            return fallback_text
        
        # Configure generation parameters
        generation_config = {
            'temperature': temperature,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                # Generate content
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract and clean the text
                generated_text = response.text.strip()
                
                if generated_text:
                    return generated_text
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check for specific error types that shouldn't retry
                if "quota" in error_msg.lower() or "api_key" in error_msg.lower():
                    break
                
                # Wait before retrying
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # Return fallback text if all attempts failed
        return fallback_text
    
    def generate_batch_insights(self, prompts_dict: dict, fallbacks_dict: dict = None) -> dict:
        """
        Generate multiple insights in batch.
        
        Args:
            prompts_dict: Dictionary of section_name -> prompt
            fallbacks_dict: Dictionary of section_name -> fallback text
        
        Returns:
            Dictionary of section_name -> generated text
        """
        results = {}
        fallbacks = fallbacks_dict or {}
        
        total = len(prompts_dict)
        current = 0
        
        for section_name, prompt in prompts_dict.items():
            current += 1
            
            fallback = fallbacks.get(section_name, f"Analysis for {section_name}")
            results[section_name] = self.generate_insight(prompt, fallback)
            
            # Small delay between requests to avoid rate limiting
            if current < total:
                time.sleep(0.5)
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not getattr(self, 'available', False) or not self.model:
            return False
        
        try:
            test_prompt = "Say 'Connection successful' in exactly three words."
            response = self.model.generate_content(test_prompt)
            
            if response and response.text:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def format_data_for_prompt(self, data, max_rows: int = 100) -> str:
        """
        Format data (DataFrame or dict) for inclusion in prompts.
        
        Args:
            data: Pandas DataFrame or dictionary to format
            max_rows: Maximum number of rows to include
        
        Returns:
            Formatted string representation of the data
        """
        try:
            # Handle pandas DataFrames
            if hasattr(data, 'to_string'):
                # Limit rows if needed
                if len(data) > max_rows:
                    data_subset = data.head(max_rows)
                    return f"{data_subset.to_string()}\n\n[Note: Showing first {max_rows} of {len(data)} total rows]"
                else:
                    return data.to_string()
            
            # Handle dictionaries
            elif isinstance(data, dict):
                formatted_lines = []
                for key, value in data.items():
                    # Handle nested DataFrames in dictionaries
                    if hasattr(value, 'to_string'):
                        formatted_lines.append(f"\n{key}:\n{value.to_string()}")
                    else:
                        formatted_lines.append(f"{key}: {value}")
                return "\n".join(formatted_lines)
            
            # Handle other types
            else:
                return str(data)
                
        except Exception as e:
            return str(data)[:5000]  # Fallback with truncation