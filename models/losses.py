import torch.nn as nn
import torchvision.models as models
###############################################################################
# Functions
###############################################################################

class ContentLoss:
	def __init__(self, loss):
		self.criterion = loss
			
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)

class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model                 #截取0~14层的VGG19赋予model
		
	def __init__(self, loss):
		self.criterion = loss  #损失函数
		self.contentFunc = self.contentFunc()	#获取截取的model
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss

def init_loss(opt, tensor):
	# disc_loss = None
	# content_loss = None
	
	if opt.model == 'pix2pix':
		perceptual_loss = PerceptualLoss(nn.MSELoss())
		# content_loss.initialize(nn.MSELoss())
	else:
		perceptual_loss = PerceptualLoss(nn.MSELoss())
		# content_loss.initialize(nn.L1Loss())
	#else:
		#raise ValueError("Model [%s] not recognized." % opt.model)
	
	#if opt.gan_type == 'wgan-gp':
		#disc_loss = DiscLossWGANGP(opt, tensor)
	#elif opt.gan_type == 'lsgan':
		#disc_loss = DiscLossLS(opt, tensor)
	#elif opt.gan_type == 'gan':
		#disc_loss = DiscLoss(opt, tensor)
	#else:
		#raise ValueError("GAN [%s] not recognized." % opt.gan_type)
	# disc_loss.initialize(opt, tensor)
	return perceptual_loss