def create_opt():
    opt = {}

    # -------------------------------------------------------------------------
    # 通用选项
    # -------------------------------------------------------------------------
    opt['cuda'] = '0'
    opt['seed'] = None

    opt['train_dir']   = 'data/processed/train/'
    opt['train_file']  = 'data/processed/train_labels.csv'
    opt['test_dir']    = 'data/processed/test/'
    opt['test_file']   = 'data/processed/submission_format.csv'
    opt['out_dir']     = 'result/'

    opt['fold']        = 10
    opt['label_dir']   = 'data/processed/10-fold/'

    # -------------------------------------------------------------------------
    # 训练选项
    # -------------------------------------------------------------------------
    opt['use_fake_label'] = True
    opt['fake_dir']       = 'save/0.9126-0.9234'

    opt['train_bs']    = 128
    opt['max_N']       = 3
    opt['lr_list']     = [1e-3, 5e-4, 1e-4, 1e-5, 1e-6]
    opt['num_workers'] = 32
    opt['num_epochs']  = 1000

    # -------------------------------------------------------------------------
    # 网络选项
    # -------------------------------------------------------------------------
    opt['net']          = 'resnet18'
    opt['optimizer']    = 'adam'
    opt['weight_decay'] = 0.
    opt['loss']         = 'cross_entropy'

    # -------------------------------------------------------------------------
    # 数据增强的选项
    # -------------------------------------------------------------------------
    opt['augment']    = True
    opt['num_repeat'] = 8

    # 水平翻转
    opt['phflip'] = 0.

    # 随机裁剪的概率
    opt['crop_probability'] = 0.
    # 裁剪后的宽度
    opt['crop_width']       = 40

    # 进行水平平移的概率
    opt['phtranslation'] = 1.
    # 向左平移的概率（向右平移的概率同过“1-”计算得到）
    opt['pltranslation'] = 0.5
    # 向左平移的最大幅度, number, left, translation
    opt['nltranslation'] = 100
    # 向右平移的最大幅度, number, right, translation
    opt['nrtranslation'] = 100

    # 开启循环平移的概率
    opt['pcircle']  = 1.
    # 向左循环平移的概率
    opt['plcircle'] = 0.5
    # 向左循环平移的一组的宽度
    opt['glcircle'] = 40
    # 向左循环平移的最大次数
    opt['nlcircle'] = 4
    # 向右循环平移的一组的宽度
    opt['grcircle'] = 40
    # 向右循环平移的最大次数
    opt['nrcircle'] = 4

    # 开启重叠平移的概率
    opt['pot']     = 1.
    # 向左重叠平移的概率
    opt['polt']    = 0.5
    # 向左重叠的最大列数
    opt['wolt']    = 100
    # 向左重叠的位置的low和high
    opt['lowolt']  = 0
    opt['higholt'] = 70
    # 向右重叠的最大列数
    opt['wort']    = 100
    # 向右重叠的位置的low和high
    opt['lowort']  = 102
    opt['highort'] = 172

    # 开启随机将某些列置0这项操作的概率, probability, vertical, invalid
    opt['pvinvalid'] = 1.
    # 随机置0的列的最大个数, number vertical, invalid
    opt['nvinvalid'] = 8
    # 每次置0的列的最大宽度, width vertical invalid
    opt['wvinvalid'] = 20

    # 开启随机将某些行置0的这项操作的概率
    opt['phinvalid'] = 1.
    # 随机置0的最大次数
    opt['nhinvalid'] = 8
    # 随机置0的最大宽度
    opt['whinvalid'] = 15

    return opt
