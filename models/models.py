
def create_model(opt):
    model = None
    print((opt.model))

    if opt.model == 'PATN':
        assert opt.dataset_mode == 'keypoint'
        from .PATN import TransferModel
        model = TransferModel()
        model.initialize(opt)
    elif opt.model == 'PATN_FUNIT':
        assert opt.dataset_mode == 'keypoint'
        from .ylb_two_stage import PatnFunitModel
        model = PatnFunitModel()
        model.initialize(opt['patn'], opt['funit'])

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    print(("model [%s] was created" % (model.name())))
    return model
