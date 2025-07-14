import os
import numpy as np
from PIL import Image
import surface_distance as surfdist
class dice():
    def __init__(self,t,p):
        self.true_mask_folder=t
        self.pred_mask_folder=p
    def calculate_dice(self,mask1, mask2):
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        intersection = np.logical_and(mask1, mask2)
        dice = 2. * intersection.sum() / (mask1.sum() + mask2.sum())
        return dice
    def calculate_iou(self,mask1,mask2):
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        return intersection.sum()/union.sum()
    def calculate_precision(self,mask1,mask2):
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        mask1[np.where(mask2==0)]=0
        return mask1.sum()/mask2.sum()
    def calculate_recall(self,mask1,mask2):
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        mask2[np.where(mask1==0)]=0
        intersection = np.logical_and(mask1, mask2)
        return mask2.sum()/mask1.sum()
    def calculate_asd(self,mask1,mask2):
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        surface_distances=surfdist.compute_surface_distances(mask1, mask2,(1.0,1.0))
        return surfdist.compute_average_surface_distance(surface_distances)
    def load_masks(self,folder):
        masks = {}
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                path = os.path.join(folder, filename)
                masks[filename] = Image.open(path).convert('1')
                
        return masks

    def do(self):
        true_masks = self.load_masks(self.true_mask_folder)
        pred_masks = self.load_masks(self.pred_mask_folder)
        dices = []
        ious=[]
        pres=[]
        recs=[]
        asds1=[]
        asds2=[]
        for filename in true_masks:
            #filename1=filename.split('.')[0]+'_pred.png'
            if filename in pred_masks:
                true_mask = true_masks[filename]
                pred_mask = pred_masks[filename]
                dice_score = self.calculate_dice(true_mask, pred_mask)*100
                iou_score = self.calculate_iou(true_mask, pred_mask)*100
                pre_score = self.calculate_precision(true_mask, pred_mask)*100
                rec_score = self.calculate_recall(true_mask, pred_mask)*100
                asd_score1,asd_score2 = self.calculate_asd(true_mask, pred_mask)
                dices.append(dice_score)
                ious.append(iou_score)
                pres.append(pre_score)
                recs.append(rec_score)
                asds1.append(asd_score1)
                asds2.append(asd_score2)
            else:
                print(f"Prediction for {filename} not found.")
        z_value=1.96
        average_dice = np.mean(dices)
        std_dice=np.std(dices)
        standard_error_dice = std_dice / (200 ** 0.5)
        bound_width_dice = 2*z_value * standard_error_dice
        variance_dice = (bound_width_dice / (2 * z_value)) ** 2
        #print('%.2f,%.2f'%(average_dice,variance_dice))
        average_iou = np.mean(ious)
        std_iou=np.std(ious)
        standard_error_iou = std_iou / (200 ** 0.5)
        bound_width = 2*z_value * standard_error_iou
        variance_iou = (bound_width / (2 * z_value)) ** 2
        #print('%.2f,%.2f'%(average_iou,variance_iou))
        average_pre = np.mean(pres)
        std_pre=np.std(pres)
        standard_error_pre = std_pre / (200 ** 0.5)
        bound_width_pre = 2*z_value * standard_error_pre
        variance_pre = (bound_width_pre / (2 * z_value)) ** 2
        #print('%.2f,%.2f'%(average_pre,variance_pre))
        average_rec = np.mean(recs)
        std_rec=np.std(recs)
        standard_error_rec = std_rec / (200 ** 0.5)
        bound_width_rec = 2*z_value * standard_error_rec
        variance_rec = (bound_width_rec / (2 * z_value)) ** 2
        #print('%.2f,%.2f'%(average_rec,variance_rec))
        average_asd1 = np.mean(asds1)
        std_asd1=np.std(asds1)
        average_asd2 = np.mean(asds2)
        std_asd2=np.std(asds2)
        average_asd=(average_asd1+average_asd2)/2
        std_asd=(std_asd1+std_asd2)/2
        standard_error_asd = std_asd / (200 ** 0.5)
        bound_width_asd = 2*z_value * standard_error_asd
        variance_asd = (bound_width_asd / (2 * z_value)) ** 2
        #print('%.2f,%.2f'%(average_asd,variance_asd))
        print('%.2f\\tiny$\\pm$%.2f& %.2f\\tiny$\\pm$%.2f& %.2f\\tiny$\\pm$%.2f& %.2f\\tiny$\\pm$%.2f& %.2f\\tiny$\\pm$%.2f\\\\'%(average_dice,variance_dice,average_iou,variance_iou,average_pre,variance_pre,average_rec,variance_rec,average_asd,variance_asd))
        txt = self.pred_mask_folder + '/metrics.txt'
        with open(txt, 'w') as f:
            f.write('dice:%.2f,%.2f\n'%(average_dice,variance_dice))
            f.write('iou:%.2f,%.2f\n'%(average_iou,variance_iou))
            f.write('pre:%.2f,%.2f\n'%(average_pre,variance_pre))
            f.write('rec:%.2f,%.2f\n'%(average_rec,variance_rec))
            f.write('asd:%.2f,%.2f\n'%(average_asd,variance_asd))
            
if __name__ == '__main__':
    true_folder="/home/jinzhuo/jzproject/U-Mamba/data/nnUNet_raw/Dataset003_Arcade/gt/"
    pred_folder="/data/predict/"
    calculater=dice(true_folder,pred_folder)
    calculater.do()