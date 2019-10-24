
def create_model(opt):
    model = None
    print((opt.model))

    if opt.model == 'PATN':
        assert opt.dataset_mode == 'keypoint'
        from .PATN import TransferModel
        model = TransferModel()
    elif opt.model == 'PATN_FUNIT':
        assert opt.dataset_mode == 'keypoint_funit'
        from .PATN_FUNIT import PatnFunitModel
        model = PatnFunitModel()
    elif opt.model == 'PATN_FUNIT_NO_GAN':
        assert opt.dataset_mode == 'keypoint_funit'
        from .PATN_FUNIT_NO_GAN import PatnFunitModel
        model = PatnFunitModel()
    elif opt.model == 'PATN_FUNIT_FULL':
        assert opt.dataset_mode == 'keypoint_funit'
        from .PATN_FUNIT_FULL import PatnFunitModel
        model = PatnFunitModel()
    elif opt.model == 'PATN_FUNIT_FULL_v2':
        assert opt.dataset_mode == 'keypoint_funit_v2', 'You must use v2 dataset with additional body part labels!'
        from .PATN_FUNIT_FULL_v2 import PatnFunitModel
        model = PatnFunitModel()
        print('PATN_FUNIT_FULL_v2: Incorporate weighted L1 loss that focus more on clothes and face')
    elif opt.model == 'PATN_FUNIT_DEBUG':
        assert opt.dataset_mode == 'keypoint_funit'
        from .PATN_FUNIT_DEBUG import PatnFunitModel
        model = PatnFunitModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    print(("model [%s] was created" % (model.name())))
    return model
