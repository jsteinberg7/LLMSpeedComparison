#!/usr/bin/env python3
"""
LLM Comparison Script

This script compares response times between OpenAI and Groq APIs for the same prompt.
It measures time to first token and total completion time.
"""

import os
import time
import argparse
import asyncio
from typing import Dict, Any, Tuple
import re


import openai
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO: Create a .env file with your API keys:
# OPENAI_API_KEY=your_openai_api_key
# GROQ_API_KEY=your_groq_api_key

# Initialize clients
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Default models
OPENAI_MODEL = "gpt-4o"
GROQ_MODEL = "llama3-70b-8192"

class TimingStats:
    def __init__(self):
        self.start_time = None
        self.first_token_time = None
        self.completion_time = None
        self.total_tokens = 0
        self.tokens_per_sec = None  # Direct field for tokens per second
        self.ttft_direct = None     # Direct field for time to first token
        self.total_time_direct = None  # Direct field for total completion time
    
    def start(self):
        self.start_time = time.time()
    
    def record_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()
    
    def finish(self, total_tokens: int):
        self.completion_time = time.time()
        self.total_tokens = total_tokens
        # Calculate tokens per second directly
        if self.total_completion_time and self.total_completion_time > 0:
            self.tokens_per_sec = total_tokens / self.total_completion_time
    
    @property
    def time_to_first_token(self) -> float:
        if self.ttft_direct is not None:
            return self.ttft_direct
        elif self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None
    
    @property
    def total_completion_time(self) -> float:
        if self.total_time_direct is not None:
            return self.total_time_direct
        elif self.completion_time and self.start_time:
            return self.completion_time - self.start_time
        return None
    
    @property
    def tokens_per_second(self) -> float:
        if self.tokens_per_sec is not None:
            return self.tokens_per_sec
        elif self.total_tokens and self.total_completion_time and self.total_completion_time > 0:
            return self.total_tokens / self.total_completion_time
        return None


def parse_prompt_file(file_content: str) -> list:
    """Parse the prompt file content into a list of message dictionaries."""
    messages = []
    
    # Handle the more complex content with potential multi-line strings and escaping
    # Each message starts with {:role=> and ends with }
    message_pattern = r'\{:role=>"([^"]+)", :content=>"((?:[^"\\]|\\.|"(?:\\.|[^"\\])*")*?)"\}'
    
    # Find all matches in the file content
    for match in re.finditer(message_pattern, file_content, re.DOTALL):
        role, content = match.groups()
        
        # Convert system/user/assistant roles to match OpenAI and Groq format
        if role in ["system", "user", "assistant"]:
            # Clean up escaped characters
            content = content.replace('\\"', '"')
            # Remove escaped backslashes
            content = content.replace('\\\\', '\\')
            # Handle escaped brackets and other special characters
            content = content.replace('\\[', '[').replace('\\]', ']')
            content = content.replace('\\"', '"')
            
            messages.append({"role": role, "content": content})
    
    # Print diagnostic information
    print(f"Parsed {len(messages)} messages from prompt file")
    for i, msg in enumerate(messages):
        role_preview = msg["role"]
        content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  Message {i+1}: role={role_preview}, content={content_preview}")
    
    return messages


async def query_openai(prompt: str, model: str = OPENAI_MODEL) -> Tuple[str, TimingStats]:
    """Query OpenAI API and track timing stats."""
    stats = TimingStats()
    full_response = ""
    
    # Parse prompt if it's in the Ruby-style hash format
    if "{:role=>" in prompt:
        messages = parse_prompt_file(prompt)
    else:
        # Fallback to simple user message if not in the expected format
        messages = [{"role": "user", "content": prompt}]
    
    stats.start()
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if not full_response:
                stats.record_first_token()
            full_response += content
    
    # Get token count from a non-streaming call to ensure accuracy
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    token_count = completion.usage.total_tokens
    
    stats.finish(token_count)
    return full_response, stats


async def query_groq(prompt: str, model: str = GROQ_MODEL) -> Tuple[str, TimingStats]:
    """Query Groq API and track timing stats."""
    stats = TimingStats()
    full_response = ""
    
    # Parse prompt if it's in the Ruby-style hash format
    if "{:role=>" in prompt:
        messages = parse_prompt_file(prompt)
    else:
        # Fallback to simple user message if not in the expected format
        messages = [{"role": "user", "content": prompt}]
    
    stats.start()
    stream = await groq_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if not full_response:
                stats.record_first_token()
            full_response += content
    
    # Get token count from a non-streaming call
    completion = await groq_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    token_count = completion.usage.total_tokens
    
    stats.finish(token_count)
    return full_response, stats


def format_stats(stats: Dict[str, TimingStats]) -> str:
    """Format the timing statistics as a readable string."""
    result = "=== Comparison Results ===\n\n"
    
    for provider, provider_stats in stats.items():
        result += f"{provider}:\n"
        
        # Safely format values, handling None cases
        ttft = provider_stats.time_to_first_token
        ttft_str = f"{ttft:.3f}" if ttft is not None else "N/A"
        
        total_time = provider_stats.total_completion_time
        total_time_str = f"{total_time:.3f}" if total_time is not None else "N/A"
        
        tps = provider_stats.tokens_per_second
        tps_str = f"{tps:.2f}" if tps is not None else "N/A"
        
        result += f"  Time to first token: {ttft_str} seconds\n"
        result += f"  Total completion time: {total_time_str} seconds\n"
        result += f"  Total tokens: {provider_stats.total_tokens}\n"
        result += f"  Tokens per second: {tps_str}\n\n"
    
    # Calculate differences
    providers = list(stats.keys())
    if len(providers) >= 2:
        p1, p2 = providers[0], providers[1]
        s1, s2 = stats[p1], stats[p2]
        
        # Only calculate differences if both values are not None
        if s1.time_to_first_token is not None and s2.time_to_first_token is not None:
            ttft_diff = s1.time_to_first_token - s2.time_to_first_token
            ttft_pct = (ttft_diff / s2.time_to_first_token) * 100 if s2.time_to_first_token else 0
            faster_ttft = p1 if ttft_diff < 0 else p2
            result += "Comparison:\n"
            result += f"  Time to first token: {faster_ttft} is {abs(ttft_diff):.3f} seconds faster ({abs(ttft_pct):.1f}%)\n"
        
        if s1.total_completion_time is not None and s2.total_completion_time is not None:
            comp_time_diff = s1.total_completion_time - s2.total_completion_time
            comp_time_pct = (comp_time_diff / s2.total_completion_time) * 100 if s2.total_completion_time else 0
            faster_comp = p1 if comp_time_diff < 0 else p2
            if "Comparison:" not in result:
                result += "Comparison:\n"
            result += f"  Total completion time: {faster_comp} is {abs(comp_time_diff):.3f} seconds faster ({abs(comp_time_pct):.1f}%)\n"
        
        if s1.tokens_per_second is not None and s2.tokens_per_second is not None:
            tokens_per_sec_diff = s1.tokens_per_second - s2.tokens_per_second
            tokens_per_sec_pct = (tokens_per_sec_diff / s2.tokens_per_second) * 100 if s2.tokens_per_second else 0
            faster_tps = p1 if tokens_per_sec_diff > 0 else p2
            if "Comparison:" not in result:
                result += "Comparison:\n"
            result += f"  Tokens per second: {faster_tps} is {abs(tokens_per_sec_diff):.2f} tokens/sec faster ({abs(tokens_per_sec_pct):.1f}%)\n"
    
    return result


async def main():
    parser = argparse.ArgumentParser(description="Compare LLM API response times.")
    parser.add_argument("--prompt-file", default="prompt.txt", help="Path to the prompt file")
    parser.add_argument("--openai-model", default=OPENAI_MODEL, help="OpenAI model to use")
    parser.add_argument("--groq-model", default=GROQ_MODEL, help="Groq model to use")
    parser.add_argument("--no-save-responses", action="store_true", help="Don't save responses to files")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per model")
    parser.add_argument("--delay", type=int, default=0, help="Delay between runs in seconds")
    args = parser.parse_args()
    
    # Read prompt from file
    try:
        with open(args.prompt_file, "r") as f:
            prompt = f.read().strip()
            if not prompt:
                print(f"Warning: The prompt file '{args.prompt_file}' is empty.")
                return
    except FileNotFoundError:
        print(f"Error: Prompt file '{args.prompt_file}' not found.")
        return
    
    print(f"Running comparison with prompt from {args.prompt_file}")
    print(f"OpenAI model: {args.openai_model}")
    print(f"Groq model: {args.groq_model}")
    print(f"Number of runs per model: {args.runs}")
    print(f"Delay between runs: {args.delay} seconds")
    print("Starting API calls...\n")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Initialize stats and responses dictionaries
    all_stats = {"OpenAI": [], "Groq": []}
    all_responses = {"OpenAI": [], "Groq": []}
    
    # Run OpenAI queries
    for run in range(1, args.runs + 1):
        try:
            print(f"Querying OpenAI (Run {run}/{args.runs})...")
            openai_response, openai_stats = await query_openai(prompt, model=args.openai_model)
            all_stats["OpenAI"].append(openai_stats)
            all_responses["OpenAI"].append(openai_response)
            print(f"OpenAI Run {run} complete ({openai_stats.total_tokens} tokens)")
            
            # Save this run's response
            if not args.no_save_responses and openai_response:
                os.makedirs("responses", exist_ok=True)
                filename = f"responses/openai_run{run}_response_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.write(openai_response)
                print(f"Saved OpenAI Run {run} response to {filename}")
                
            # Wait between runs, but not after the last run
            if run < args.runs:
                print(f"Waiting {args.delay} seconds before next run...")
                await asyncio.sleep(args.delay)
                
        except Exception as e:
            print(f"Error querying OpenAI (Run {run}): {e}")
    
    # Run Groq queries
    for run in range(1, args.runs + 1):
        try:
            print(f"Querying Groq (Run {run}/{args.runs})...")
            groq_response, groq_stats = await query_groq(prompt, model=args.groq_model)
            all_stats["Groq"].append(groq_stats)
            all_responses["Groq"].append(groq_response)
            print(f"Groq Run {run} complete ({groq_stats.total_tokens} tokens)")
            
            # Save this run's response
            if not args.no_save_responses and groq_response:
                os.makedirs("responses", exist_ok=True)
                filename = f"responses/groq_run{run}_response_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.write(groq_response)
                print(f"Saved Groq Run {run} response to {filename}")
                
            # Wait between runs, but not after the last run
            if run < args.runs:
                print(f"Waiting {args.delay} seconds before next run...")
                await asyncio.sleep(args.delay)
                
        except Exception as e:
            print(f"Error querying Groq (Run {run}): {e}")
    
    # Calculate average stats
    avg_stats = {}
    
    for provider, stats_list in all_stats.items():
        if not stats_list:  # Skip if no successful runs
            continue
            
        # Create a new TimingStats object for averages
        avg_stat = TimingStats()
        
        # Calculate time to first token average directly
        valid_ttft_values = [s.time_to_first_token for s in stats_list if s.time_to_first_token is not None]
        if valid_ttft_values:
            # Set the direct property instead of trying to calculate it
            avg_stat.ttft_direct = sum(valid_ttft_values) / len(valid_ttft_values)
        
        # Calculate total completion time average directly  
        valid_completion_times = [s.total_completion_time for s in stats_list if s.total_completion_time is not None]
        if valid_completion_times:
            # Set the direct property
            avg_stat.total_time_direct = sum(valid_completion_times) / len(valid_completion_times)
        
        # Calculate token count average
        avg_stat.total_tokens = sum(s.total_tokens for s in stats_list) / len(stats_list)
        
        # Calculate tokens per second average directly
        valid_tps_values = [s.tokens_per_second for s in stats_list if s.tokens_per_second is not None]
        if valid_tps_values:
            avg_stat.tokens_per_sec = sum(valid_tps_values) / len(valid_tps_values)
        
        # Add calculated averages to stats dictionary
        avg_stats[provider] = avg_stat
    
    # Print and save individual run stats
    os.makedirs("results", exist_ok=True)
    stats_filename = f"results/comparison_stats_{timestamp}.txt"
    
    with open(stats_filename, "w") as f:
        # Write header
        header = f"=== Comparison Results ({args.runs} runs per model) ===\n\n"
        f.write(header)
        print("\n" + header)
        
        # Write individual run stats
        for provider, stats_list in all_stats.items():
            if not stats_list:
                continue
                
            provider_header = f"{provider} - Individual Runs:\n"
            f.write(provider_header)
            print(provider_header)
            
            for i, stat in enumerate(stats_list):
                run_info = f"  Run {i+1}:\n"
                
                # Safely format values, handling None cases
                ttft = stat.time_to_first_token
                ttft_str = f"{ttft:.3f}" if ttft is not None else "N/A"
                
                total_time = stat.total_completion_time
                total_time_str = f"{total_time:.3f}" if total_time is not None else "N/A"
                
                tps = stat.tokens_per_second
                tps_str = f"{tps:.2f}" if tps is not None else "N/A"
                
                run_info += f"    Time to first token: {ttft_str} seconds\n"
                run_info += f"    Total completion time: {total_time_str} seconds\n"
                run_info += f"    Total tokens: {stat.total_tokens}\n"
                run_info += f"    Tokens per second: {tps_str}\n\n"
                
                f.write(run_info)
                print(run_info)
        
        # Write average stats
        avg_stats_text = format_stats(avg_stats)
        avg_header = "Average Stats Across All Runs:\n"
        f.write("\n" + avg_header + avg_stats_text)
        print(avg_header + avg_stats_text)
    
    print(f"Saved detailed comparison stats to {stats_filename}")


if __name__ == "__main__":
    asyncio.run(main()) 