k_n_layers=1
k_hid_size=128
k_out_size=128
epochs=200

for k_n_layers in 1 3 5 
do 
    for k_hid_size in 128 
    do
        for k_out_size in 16 64 128 
        do
            tag=n-${k_n_layers}_h-${k_hid_size}_o-${k_out_size}
            echo $tag
            ./run.sh \
                --tag $tag \
                --k_n_layers $k_n_layers  \
                --k_hid_size $k_hid_size  \
                --k_out_size $k_out_size  \
                --epochs $epochs
        done
    done
done