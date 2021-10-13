# ======== Office-31 =========
# source training
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office31/r0/src/ --dset office --max_epoch 100 --s 0
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office31/r0/src/ --dset office --max_epoch 100 --s 1
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office31/r0/src/ --dset office --max_epoch 100 --s 2

# target training
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 0 --t 1  --output_src Office31/r0/src/ --output Office31/r0/sclm/  --dset office --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 0 --t 2  --output_src Office31/r0/src/ --output Office31/r0/sclm/  --dset office --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 1 --t 0  --output_src Office31/r0/src/ --output Office31/r0/sclm/  --dset office --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 1 --t 2  --output_src Office31/r0/src/ --output Office31/r0/sclm/  --dset office --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 2 --t 0  --output_src Office31/r0/src/ --output Office31/r0/sclm/  --dset office --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 2 --t 1  --output_src Office31/r0/src/ --output Office31/r0/sclm/  --dset office --lr 1e-2 --net resnet50


# ======= Office-Home ========
# source training
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office-Home/r0/src/ --dset office-home --max_epoch 100 --s 0
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office-Home/r0/src/ --dset office-home --max_epoch 100 --s 1
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office-Home/r0/src/ --dset office-home --max_epoch 100 --s 2
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Office-Home/r0/src/ --dset office-home --max_epoch 100 --s 3

# target training
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 0 --t 1  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 0 --t 2  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 0 --t 3  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 1 --t 0  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 1 --t 2  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 1 --t 3  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 2 --t 0  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 2 --t 1  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 2 --t 3  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 3 --t 0  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 3 --t 1  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 3 --t 2  --output_src Office-Home/r0/src/ --output Office-Home/r0/sclm/  --dset office-home --lr 1e-2 --net resnet50


# ========== VisDA-C =========
# source training
python SCLM_source.py --trte val --da uda --gpu_id 0  --output Visda/r0/src/ --dset VISDA-C --max_epoch 10 --s 0 --lr 1e-3 --net resnet101

# target training
python SCLM_target.py --da uda --gpu_id 0 --cls_par 0.3 --cls_snt 0.1 --s 0 --t 1  --output_src Visda/r0/src/ --output Visda/r0/sclm/  --dset VISDA-C --lr 1e-3 --net resnet101