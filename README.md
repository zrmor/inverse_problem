## Setup

This project requires a Weights & Biases API key for logging experiments. You can set it up in one of two ways:

1. Create a `.env` file in the project root and add:
   ```
   WANDB_API_KEY=your_api_key_here
   ```

2. Or set it as an environment variable in your terminal:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

If neither is set, you'll be prompted to enter your API key when running the notebook.