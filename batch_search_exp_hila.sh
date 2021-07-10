# Mean teacher; augmentation: translation and horizontal flips
#python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp_ctaa/log_mnist_usps_ctaa_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp_ctaa/log_mnist_usps_ctaa_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=30 --epoch_size=large --combine_batches=true --device=${1}
