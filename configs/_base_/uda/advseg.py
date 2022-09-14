# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# uda settings
uda = dict(
    type='AdvSeg',
    discriminator_type='LS',
    lr_D=1e-4,
    lr_D_power=0.9,
    lr_D_min=0,
    lambda_adv_target=dict(main=0.001, aux=0.0002),
    debug_img_interval=1000)
