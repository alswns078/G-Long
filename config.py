import argparse

def get_args(description='Config of G-Long (MSC)'):
    parser = argparse.ArgumentParser(description=description)

    # Logging
    parser.add_argument("--log_name", type=str, default="eval_log.json")
    
    # Data Configuration
    parser.add_argument("--dataset", choices=["msc"], default="msc", help="Dataset to use.")
    parser.add_argument('--data_path', type=str, default='data/MSC', help='Dataset directory.')
    parser.add_argument('--data_name', type=str, default='seq_test_part_0.json', help='Dataset filename.')

    # Memory Settings
    parser.add_argument("--memory_cache", nargs='?', default=None, help="Path to ChromaDB cache.")
    parser.add_argument('--relevance_memory_number', type=int, default=3, help='Top-k retrieved memories.')
    parser.add_argument('--context_memory_number', type=int, default=30, help='Recent context window size.')

    # Triplet Extraction (sLM vs LLM)
    parser.add_argument('--triplet_mode', type=str, choices=['llm', 'slm'], default='slm', 
                        help='Extraction mode. "slm" uses pre-extracted JSON for efficiency.')
    parser.add_argument('--slm_data_path', type=str, default='data/MSC/qwen_msc_triplet.json', 
                        help='Path to sLM triplet file.')

    # Client Settings (OpenAI Only)
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Main backbone model.')
    parser.add_argument('--api_key', type=str, default='', help='OpenAI API Key.')

    # Evaluation Settings
    parser.add_argument('--usr_name', type=str, default='User')
    parser.add_argument('--agent_name', type=str, default='Agent')
    parser.add_argument('--test_num', type=int, default=0, help='0 indicates testing all samples.')
    parser.add_argument('--build_times', type=int, default=1)
    parser.add_argument('--min_session', type=int, default=1)
    parser.add_argument('--max_session', type=int, default=5)
    parser.add_argument('--log_step', type=int, default=10)

    args = parser.parse_args()
    return args