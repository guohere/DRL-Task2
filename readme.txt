


## "Mature" Problem

* Summarization with RLHF (PPO)
Link: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf

* Helpful Chatbot


## minicode 

Link: https://github.com/ttumiel/minRLHF/tree/master

To make runnable code:

Fixed shape mismatches in forward() and generate() functions in model.py by ensuring correct tensor reshaping.

Implemented sequence trimming to prevent exceeding block_size.

Optimized CUDA memory usage by clearing cache and enabling expandable_segments.

Limited dataset to 5000 samples (out of approx. 130,000) to reduce training time.

Adjusted training epochs and iterations.

Added debugging print statements for easier error tracking.