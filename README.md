# LLM Response Time Comparison

This tool compares response times between OpenAI and Groq APIs by measuring:
- Time to first token (latency)
- Total completion time
- Tokens per second (throughput)

It performs multiple runs for more accurate comparisons, calculates averages, and automatically saves all outputs and statistics to files.

## Setup

1. **Install dependencies**:
   ```bash
   pip install openai groq python-dotenv
   ```

2. **Create a `.env` file** in the same directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Prepare your prompt** in the `prompt.txt` file.

   The script supports two prompt formats:
   
   - **Simple text**: Plain text prompts will be sent as user messages
   
   - **Conversation format**: Ruby-style hash format for full conversations with multiple roles
     ```
     {:role=>"system", :content=>"You are a helpful assistant."}
     {:role=>"user", :content=>"Hello, how are you?"}
     {:role=>"assistant", :content=>"I'm doing well, thank you for asking!"}
     {:role=>"user", :content=>"Can you help me with a question?"}
     ```
     This format allows you to test complex multi-turn conversations with system prompts.

## Usage

Run the script with default settings:
```bash
python llm_comparison.py
```

### Advanced Options

- **Use a different prompt file**:
  ```bash
  python llm_comparison.py --prompt-file custom_prompt.txt
  ```

- **Choose different models**:
  ```bash
  python llm_comparison.py --openai-model gpt-4-turbo --groq-model llama2-70b-4096
  ```

- **Don't save responses to files**:
  ```bash
  python llm_comparison.py --no-save-responses
  ```

- **Change the number of runs per model**:
  ```bash
  python llm_comparison.py --runs 5
  ```

- **Change the delay between runs (in seconds)**:
  ```bash
  python llm_comparison.py --delay 30
  ```

## Default Models

- OpenAI: `gpt-4o`
- Groq: `llama3-70b-8192`

## Output Files

By default, the script saves:
- Individual model responses for each run in the `responses/` directory
- Comparison statistics with individual run data and averages in the `results/` directory

## Examples

Example output (with 3 runs per model):
```
=== Comparison Results (3 runs per model) ===

OpenAI - Individual Runs:
  Run 1:
    Time to first token: 0.542 seconds
    Total completion time: 3.218 seconds
    Total tokens: 156
    Tokens per second: 48.48

  Run 2:
    Time to first token: 0.537 seconds
    Total completion time: 3.185 seconds
    Total tokens: 158
    Tokens per second: 49.61

  Run 3:
    Time to first token: 0.561 seconds
    Total completion time: 3.294 seconds
    Total tokens: 157
    Tokens per second: 47.66

Groq - Individual Runs:
  Run 1:
    Time to first token: 0.324 seconds
    Total completion time: 2.654 seconds
    Total tokens: 142
    Tokens per second: 53.50

  Run 2:
    Time to first token: 0.315 seconds
    Total completion time: 2.589 seconds
    Total tokens: 141
    Tokens per second: 54.46

  Run 3:
    Time to first token: 0.341 seconds
    Total completion time: 2.714 seconds
    Total tokens: 143
    Tokens per second: 52.69

Average Stats Across All Runs:
=== Comparison Results ===

OpenAI:
  Time to first token: 0.547 seconds
  Total completion time: 3.232 seconds
  Total tokens: 157.0
  Tokens per second: 48.58

Groq:
  Time to first token: 0.327 seconds
  Total completion time: 2.652 seconds
  Total tokens: 142.0
  Tokens per second: 53.55

Comparison:
  Time to first token: Groq is 0.220 seconds faster (67.3%)
  Total completion time: Groq is 0.580 seconds faster (21.9%)
  Tokens per second: Groq is 4.97 tokens/sec faster (10.2%)
``` 