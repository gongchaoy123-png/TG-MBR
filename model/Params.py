import argparse


def ParseArgs():
	parser = argparse.ArgumentParser(description='TARGET multi-behavior BPR recommendation')
	# dataset
	parser.add_argument('--path', default='./datasets/', type=str, help='path to datasets')
	parser.add_argument('--dataset', default='Tmall', type=str,
						help='Tmall / Taobao / Beibei / IJCAI / IJCAI_15 / Yelp / yelp_tiny / MultiInt-ML10M / retail / retail_small')

	# model
	parser.add_argument('--hidden_dim', default=32, type=int, help='embedding dimension')
	parser.add_argument('--gnn_layer', default=3, type=int, help='number of GNN layers')
	parser.add_argument('--behavior_fusion', choices=['importance', 'attention', 'equal'], default='importance',
						help='importance: learned global behavior importance; equal: uniform weights; attention: legacy alias for importance')
	parser.add_argument('--interaction_filter', action='store_true', default=False,
						help='enable candidate-level pseudo-interaction filtering (CPF)')
	parser.add_argument('--interaction_filter_strength', default=0.2, type=float,
						help='target strength of the candidate residual')
	parser.add_argument('--interaction_filter_warmup', default=0, type=int,
						help='warm-up epochs before candidate residuals start')
	parser.add_argument('--interaction_filter_ramp', default=1, type=int,
						help='epochs used to ramp the candidate residual to its target strength')
	parser.add_argument('--interaction_filter_diagnostics', action='store_true',
						help='log gradient norms and compare scores with and without the residual')
	parser.add_argument('--device', default='gpu', type=str, help='gpu or cpu')

	# training
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate (grid: 1e-2/1e-3/1e-4)')
	parser.add_argument('--batch', default=8192, type=int, help='training batch size')
	parser.add_argument('--eval_batch', default=512, type=int,
						help='number of users evaluated per chunk to control GPU memory')
	parser.add_argument('--epoch', default=100, type=int, help='epochs')
	parser.add_argument('--reg', default=1e-4, type=float, help='lambda3, L2 regularization (grid: 1e-2..1e-4)')
	parser.add_argument('--patience', default=50, type=int, help='early stop patience')
	parser.add_argument('--seed', default=42, type=int, help='random seed')
	parser.add_argument('--save_best', default=None, type=str,
						help='optional checkpoint file for the best epoch (test_early_stop protocol)')
	parser.add_argument('--eval_sample_seed', default=3407, type=int,
						help='seed for validation holdout and evaluation negatives; use 3407 to match pure GCIB')
	parser.add_argument('--eval_protocol',
						choices=['test_early_stop', 'fixed', 'valid'],
						default='test_early_stop',
						help='test_early_stop: old protocol using test for early stop; fixed: last epoch; valid: validation early stop')
	parser.add_argument('--denoise_mode', choices=['none', 'user', 'item', 'both'],
						default='both', help='representation denoising: user only, item only, both, or none')
	parser.add_argument('--denoise_gate_floor', default=0.5, type=float,
						help='representation denoising gate lower bound; g = floor + (1-floor)*sigmoid')

	return parser.parse_args()


args = ParseArgs()
