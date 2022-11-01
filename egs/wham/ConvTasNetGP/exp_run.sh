epochs=200
suffix=_3src_vae
B=8
R=3
# loss=ld_src+znorm
# loss=nll+znorm+mask
loss=ld_src

for mode in min01pct
do 
    for k_n_layers in 3
    do 
        for k_hid_size in 64
        do
            for k_out_size in 16
            do
                # for loss in nll+znorm ld_mix+ld_src+nll+znorm+mask ld_mix+ld_src+znorm
                for loss in nll+vae+kl
                # for loss in nll+znorm
                do

                    tag=${mode}_n-${k_n_layers}_h-${k_hid_size}_o-${k_out_size}_${loss}_b${B}r${R}${suffix}
                    echo $tag
                    ./run.sh \
                        --tag $tag \
                        --k_n_layers $k_n_layers  \
                        --k_hid_size $k_hid_size  \
                        --k_out_size $k_out_size  \
                        --loss $loss \
                        --mode $mode \
                        --n_blocks $B \
                        --n_repeats $R \
                        --epochs $epochs

                done
            done
        done
    done
done