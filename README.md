# Search and Rescue MARL Project
A Multi-Agent Reinforcement Learning system using TorchRL for search and rescue tasks.

## Setup
1. Clone the repository: `git clone https://github.com/elte-collective-intelligence/student-search`
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/train.py`

## Usage
- Configure experiments in `configs/`.
- Run with Hydra: `python src/train.py +experiment=exp`
- View logs in `logs/` (TensorBoard/WandB) and GIFs in `outputs/gifs/`.

## Collaboration
- [Rafig Babayev]: Environment, training logic
- [Nazrin Ibrahimli]: Visualization, testing