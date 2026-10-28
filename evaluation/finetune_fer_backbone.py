"""Fine-tune late CNN layers on clean train splits; select once on FER/RAF validation.

Run with the isolated CUDA environment, never the web application's environment.
No test labels are used for fitting, checkpoint selection or mixture selection.
"""
from __future__ import annotations
import argparse,copy,hashlib,io,json,math,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import onnx
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
from evaluation.fer_torch_graph import convert_trainable
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from evaluation.fer_benchmark import metrics
from fer_model import LABELS
OUT=ROOT/'outputs/fer-finetune-v6'
DATA=ROOT/'.runtime/fer2013-benchmark-data'
RAF=ROOT/'.runtime/rafdb-reference'
MAP={1:'surprise',2:'fear',3:'disgust',4:'happy',5:'sad',6:'angry',7:'neutral'}
MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]


class Faces(Dataset):
    def __init__(self,records,train=False,raw_output=False,gpu_augmentation=False):
        self.records=records;self.train=train;self.raw_output=raw_output;self.gpu_augmentation=gpu_augmentation
        self.augment=T.Compose([T.RandomHorizontalFlip(),T.RandomAffine(degrees=8,translate=(.025,.025),scale=(.97,1.03),interpolation=T.InterpolationMode.BILINEAR,fill=127),T.ColorJitter(brightness=.12,contrast=.12,saturation=.1),T.RandomGrayscale(p=.3)])
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        encoded,label,domain=self.records[index]
        with Image.open(io.BytesIO(encoded)) as image:pixels=cv2.resize(np.asarray(image.convert('RGB')),(260,260))
        if self.train and not self.gpu_augmentation:pixels=np.asarray(self.augment(Image.fromarray(pixels)))
        tensor=torch.from_numpy(np.array(pixels,copy=True)).permute(2,0,1)
        if self.raw_output:return tensor,label,domain
        tensor=tensor.float()/255.
        return (tensor-MEAN)/STD,label,domain


def make_data():
    ft=np.load(ROOT/'outputs/fer-efficient-v2/train_features.npz',allow_pickle=False)
    fp=np.load(ROOT/'outputs/fer-efficient-v2/publicTest_features.npz',allow_pickle=False)
    fv=np.load(ROOT/'outputs/fer-efficient-v2/privateTest_features.npz',allow_pickle=False)
    rt=np.load(ROOT/'outputs/fer-multidomain-v2/raf-train-features.npz',allow_pickle=False)
    re=np.load(ROOT/'outputs/fer-multidomain-v2/raf-test-features.npz',allow_pickle=False)
    base=np.load(ROOT/'.runtime/fer-cnn-training-validation-base.npz',allow_pickle=False)
    fold=np.load(ROOT/'outputs/fer-multidomain-v2/raf-fold.npz',allow_pickle=False)
    fh,rh=ft['hashes'],rt['hashes'];valid=fold['valid'];train=fold['train']
    forbidden=set(re['hashes'])|set(fp['hashes'])|set(fv['hashes']);valid_hashes=set(rh[valid]);seen=set();fi=[];ri=[]
    for i,h in enumerate(fh):
        if h not in forbidden and h not in valid_hashes and h not in seen:fi.append(i);seen.add(h)
    for i in train:
        h=rh[i]
        if h not in forbidden and h not in seen:ri.append(i);seen.add(h)
    fer_rows=pq.read_table(DATA/'train.parquet').to_pylist();raf_rows=pq.read_table(RAF/'train.parquet').to_pylist();public_rows=pq.read_table(DATA/'publicTest.parquet').to_pylist()
    fer_indices=ft['ids'][fi]
    records=[(fer_rows[int(i)]['image']['bytes'],int(fer_rows[int(i)]['label']),0) for i in fer_indices]
    records.extend((raf_rows[int(i)]['image']['bytes'],LABELS.index(MAP[int(raf_rows[int(i)]['label'])]),1) for i in ri)
    fer_valid=[]
    for path in base['fer_ids']:
        i=int(Path(str(path)).stem);row=public_rows[i];fer_valid.append((row['image']['bytes'],int(row['label']),0))
    keep=np.array([h not in forbidden for h in rh[valid]])
    raf_valid=[(raf_rows[int(i)]['image']['bytes'],LABELS.index(MAP[int(raf_rows[int(i)]['label'])]),1) for i in valid[keep]]
    assert np.array_equal(base['raf_ids'],valid)
    old_f=base['fer_probabilities'];old_r=base['raf_probabilities'][keep]
    # The teacher/initial head is trained outside all validation images.
    initial=OUT/'initial-head.json'
    if initial.exists():head=json.loads(initial.read_text())
    else:
        x=np.concatenate([ft['features'][fi],rt['features'][ri]])
        y=np.concatenate([ft['labels'][fi],rt['labels'][ri]])
        sample_weight=np.concatenate([np.ones(len(fi)),np.full(len(ri),len(fi)/len(ri))])
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.001,max_iter=900,solver='lbfgs'))
        classifier.fit(x,y,logisticregression__sample_weight=sample_weight)
        scaler,model=classifier[0],classifier[-1]
        head={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficients':model.coef_.tolist(),'intercept':model.intercept_.tolist()}
        initial.write_text(json.dumps(head,indent=2))
    info={'fer_train_rows':len(fi),'raf_train_rows':len(ri),'fer_public_rows':len(fer_valid),'raf_validation_rows':len(raf_valid),'train_and_validation_pixel_disjoint':True,'test_pixels_not_loaded_for_training':True,'source_camera_sha256':hashlib.sha256((ROOT/'.models/fer2013/fer-camera.onnx').read_bytes()).hexdigest()}
    return records,fer_valid,raf_valid,old_f,old_r,head,info


def initial_model(head,optimized=False):
    snapshot=OUT/('initial-static.onnx' if optimized else 'initial.onnx')
    if snapshot.exists():return convert_trainable(onnx.load(str(snapshot)),static_padding=optimized).eval()
    model=onnx.load(str(ROOT/'.models/fer2013/fer-camera.onnx'))
    weights=np.asarray(head['coefficients'])/np.asarray(head['scale']);bias=np.asarray(head['intercept'])-weights@np.asarray(head['mean'])
    replacements={'camera.weight':weights.astype(np.float32),'camera.bias':bias.astype(np.float32)}
    for i,value in enumerate(model.graph.initializer):
        if value.name in replacements:model.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(replacements[value.name],value.name))
    onnx.checker.check_model(model)
    onnx.save(model,str(OUT/'initial.onnx'))
    return convert_trainable(model).eval()


_device_constants={}
def prepare_batch(images,device,augment=False,channels_last=False):
    images=images.to(device,non_blocking=True).float()/255.
    key=str(device)
    if key not in _device_constants:_device_constants[key]=(MEAN.to(device)[None],STD.to(device)[None],torch.tensor([.2989,.587,.114],device=device)[None,:,None,None])
    mean,std,luma=_device_constants[key]
    if augment:
        n=len(images);angle=(torch.rand(n,device=device)-.5)*(16*math.pi/180);scale=.97+.06*torch.rand(n,device=device);flip=torch.where(torch.rand(n,device=device)<.5,-1.,1.)
        cs=angle.cos()/scale;sn=angle.sin()/scale;theta=torch.zeros(n,2,3,device=device)
        theta[:,0,0]=cs*flip;theta[:,0,1]=-sn;theta[:,1,0]=sn*flip;theta[:,1,1]=cs;theta[:,:,2]=(torch.rand(n,2,device=device)-.5)*.1
        grid=nn.functional.affine_grid(theta,images.shape,align_corners=False)
        images=nn.functional.grid_sample(images-.5,grid,mode='bilinear',padding_mode='zeros',align_corners=False)+.5
        brightness=.88+.24*torch.rand(n,1,1,1,device=device);contrast=.88+.24*torch.rand(n,1,1,1,device=device);saturation=.9+.2*torch.rand(n,1,1,1,device=device)
        center=(images*luma).sum(1,keepdim=True).mean((2,3),keepdim=True)
        images=((images-center)*contrast+center)*brightness
        gray=(images*luma).sum(1,keepdim=True);images=gray+(images-gray)*saturation
        images=torch.where((torch.rand(n,1,1,1,device=device)<.3),gray,images).clamp(0,1)
    images=(images-mean)/std
    return images.contiguous(memory_format=torch.channels_last) if channels_last else images


@torch.no_grad()
def predict(model,loader,device,gpu_preprocess=False,channels_last=False):
    result=[]
    model.eval()
    for images,_,_ in loader:
        images=prepare_batch(images,device,False,channels_last) if gpu_preprocess else images.to(device,non_blocking=True)
        if channels_last:images=images.contiguous(memory_format=torch.channels_last)
        logits=model(torch.cat([images,images.flip(-1)]))
        if isinstance(logits,(tuple,list)):logits=logits[0]
        n=len(images);p=((logits[:n]+logits[n:])/2).float().softmax(1)
        result.append(p.cpu().numpy())
    return np.concatenate(result)


def measure(y,p):return metrics([(LABELS[int(a)],b) for a,b in zip(y,p)],0,0)


def choose_mixture(ef,er,old_f,old_r,fy,ry,bf,br):
    candidates=[]
    for weight in (.4,.6,.8,1.):
        mf=measure(fy,(1-weight)*old_f+weight*ef);mr=measure(ry,(1-weight)*old_r+weight*er)
        eligible=mf['accuracy_overall']>=bf['accuracy_overall']+.003 and mf['macro_f1']>=bf['macro_f1']+.003 and mr['accuracy_overall']>=br['accuracy_overall']+.003 and mr['macro_f1']>=br['macro_f1']+.003
        score=min(mf['accuracy_overall']/bf['accuracy_overall'],mr['accuracy_overall']/br['accuracy_overall'])+.1*(mf['macro_f1']/bf['macro_f1']+mr['macro_f1']/br['macro_f1'])/2
        candidates.append({'weight':weight,'fer':mf,'raf':mr,'eligible':eligible,'score':score})
    return candidates,max(candidates,key=lambda c:(int(c['eligible']),c['score']))


def main(args):
    OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert torch.cuda.is_available(),'CUDA is required for this recipe'
    torch.cuda.set_per_process_memory_fraction(.55)
    torch.backends.cudnn.benchmark=True
    records,fer_valid,raf_valid,old_f,old_r,head,info=make_data()
    info.update(seed=42,epochs=args.epochs,batch=args.batch,workers=args.workers,accumulation=args.accumulation,trainable_last_parameter_tensors=80,lr_backbone=5e-5,lr_head=5e-4,grayscale_augmentation=.3,label_smoothing=.05,raf_distillation_weight=.2)
    info.update(static_padding=args.optimized,gpu_preprocess=args.gpu_preprocess)
    if args.gpu_preprocess:info['augmentation_implementation']='Per-image batched GPU affine/flip/color/grayscale; same bounds, not bitwise-identical PIL augmentation'
    (OUT/'recipe.json').write_text(json.dumps(info,indent=2));print('DATA',json.dumps(info),flush=True)
    student=initial_model(head,args.optimized).cuda()
    if args.optimized:student=student.to(memory_format=torch.channels_last)
    teacher=copy.deepcopy(student).eval()
    for p in teacher.parameters():p.requires_grad=False
    prior_epochs=0
    if args.resume_weights:
        checkpoint=torch.load(args.resume_weights,map_location='cpu',weights_only=True);parameters=dict(student.named_parameters())
        assert all(k in checkpoint['state_dict'] and checkpoint['state_dict'][k].shape==v.shape for k,v in parameters.items())
        student.load_state_dict({k:checkpoint['state_dict'][k] for k in parameters},strict=False);prior_epochs=int(checkpoint['epoch'])
        info.update(resume_checkpoint_sha256=hashlib.sha256(Path(args.resume_weights).read_bytes()).hexdigest(),prior_epochs=prior_epochs)
        (OUT/'recipe.json').write_text(json.dumps(info,indent=2))
    named=list(student.named_parameters())
    for _,p in named:p.requires_grad=False
    for _,p in named[-80:]:p.requires_grad=True
    last={id(p) for _,p in named[-2:]}
    backbone=[p for _,p in named if p.requires_grad and id(p) not in last];classifier=[p for _,p in named if id(p) in last]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':5e-5},{'params':classifier,'lr':5e-4}],weight_decay=1e-4,fused=True)
    ema=copy.deepcopy(student).eval()
    for p in ema.parameters():p.requires_grad=False
    labels=np.array([r[1] for r in records]);domains=np.array([r[2] for r in records]);sample_weights=np.zeros(len(records))
    for domain in (0,1):
        mask=domains==domain;counts=np.bincount(labels[mask],minlength=7);w=(len(labels[mask])/np.maximum(counts[labels[mask]],1))**.25;sample_weights[mask]=w/w.sum()
    generator=torch.Generator().manual_seed(42)
    sampler=WeightedRandomSampler(torch.as_tensor(sample_weights,dtype=torch.double),num_samples=len(records),replacement=True,generator=generator)
    train_loader=DataLoader(Faces(records,True,args.gpu_preprocess,args.gpu_preprocess),batch_size=args.batch,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,drop_last=True)
    floader=DataLoader(Faces(fer_valid,raw_output=args.gpu_preprocess),batch_size=min(args.batch,32),num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    rloader=DataLoader(Faces(raf_valid,raw_output=args.gpu_preprocess),batch_size=min(args.batch,32),num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    fy=np.array([r[1] for r in fer_valid]);ry=np.array([r[1] for r in raf_valid])
    bf=measure(fy,old_f)
    first_f=predict(teacher,floader,'cuda',args.gpu_preprocess,args.optimized);first_r=predict(teacher,rloader,'cuda',args.gpu_preprocess,args.optimized);camera_r=.4*old_r+.6*first_r;br=measure(ry,camera_r)
    print('INITIAL',json.dumps({'fer_expert':measure(fy,first_f)['accuracy_overall'],'raf_expert':measure(ry,first_r)['accuracy_overall'],'fer_old':bf['accuracy_overall'],'raf_camera_old':br['accuracy_overall']}),flush=True)
    scaler=torch.amp.GradScaler('cuda');criterion=nn.CrossEntropyLoss(label_smoothing=.05)
    updates_per_epoch=math.ceil(len(train_loader)/args.accumulation);total_updates=args.epochs*updates_per_epoch;update=0
    history=[];best_key=(-1.,-1.);best=None;stale=0
    if args.resume_weights:
        ef=predict(ema,floader,'cuda',args.gpu_preprocess,args.optimized);er=predict(ema,rloader,'cuda',args.gpu_preprocess,args.optimized)
        candidates,chosen=choose_mixture(ef,er,old_f,old_r,fy,ry,bf,br)
        best_key=(float(chosen['eligible']),chosen['score']);best={'epoch':prior_epochs,'loss':None,'candidates':candidates,'selected':chosen};history.append(best)
        torch.save({'state_dict':ema.state_dict(),'epoch':prior_epochs,'weight':chosen['weight'],'recipe':info},OUT/'checkpoints/best.pt')
        np.savez_compressed(OUT/'best-validation.npz',fer_expert=ef,raf_expert=er,fer_old=old_f,raf_old=old_r,fer_labels=fy,raf_labels=ry)
    for epoch in range(1,args.epochs+1):
        student.train();optimizer.zero_grad(set_to_none=True);running=0.;start=time.perf_counter()
        for step,(images,target,domain) in enumerate(train_loader,1):
            images=prepare_batch(images,'cuda',True,args.optimized) if args.gpu_preprocess else images.cuda(non_blocking=True)
            if args.optimized:images=images.contiguous(memory_format=torch.channels_last)
            target=target.cuda(non_blocking=True);domain=domain.cuda(non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits=student(images);loss=criterion(logits,target)
                mask=domain==1
                if mask.any():
                    with torch.no_grad():reference=teacher(images[mask].contiguous(memory_format=torch.channels_last) if args.optimized else images[mask])
                    kd=nn.functional.kl_div(nn.functional.log_softmax(logits[mask].float()/2,1),nn.functional.softmax(reference.float()/2,1),reduction='batchmean')*4
                    loss=loss+.2*kd
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            scaler.scale(loss/args.accumulation).backward();running+=float(loss.detach())
            if step%args.accumulation==0 or step==len(train_loader):
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                update+=1;warmup=max(1,int(total_updates*.05));factor=min(1.,update/warmup)*(.1+.9*.5*(1+math.cos(math.pi*update/total_updates)))
                optimizer.param_groups[0]['lr']=5e-5*factor;optimizer.param_groups[1]['lr']=5e-4*factor
                with torch.no_grad():
                    for ep,(_,p) in zip(ema.parameters(),named):
                        if p.requires_grad:ep.mul_(.995).add_(p,alpha=.005)
            if step%50==0:print(f'EPOCH {epoch+prior_epochs} BATCH {step}/{len(train_loader)} loss={running/step:.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        ef=predict(ema,floader,'cuda',args.gpu_preprocess,args.optimized);er=predict(ema,rloader,'cuda',args.gpu_preprocess,args.optimized)
        candidates,chosen=choose_mixture(ef,er,old_f,old_r,fy,ry,bf,br);key=(float(chosen['eligible']),chosen['score'])
        entry={'epoch':epoch+prior_epochs,'loss':running/len(train_loader),'seconds':time.perf_counter()-start,'candidates':candidates,'selected':chosen};history.append(entry)
        if key>best_key:
            best_key=key;best=entry;stale=0
            torch.save({'state_dict':ema.state_dict(),'epoch':epoch+prior_epochs,'weight':chosen['weight'],'recipe':info},OUT/'checkpoints/best.pt')
            np.savez_compressed(OUT/'best-validation.npz',fer_expert=ef,raf_expert=er,fer_old=old_f,raf_old=old_r,fer_labels=fy,raf_labels=ry)
        else:stale+=1
        (OUT/'training.json').write_text(json.dumps({'recipe':info,'baseline_fer':bf,'baseline_raf_camera':br,'history':history,'best':best},indent=2))
        print('VALIDATION',epoch+prior_epochs,json.dumps({'weight':chosen['weight'],'fer_accuracy':chosen['fer']['accuracy_overall'],'fer_f1':chosen['fer']['macro_f1'],'raf_accuracy':chosen['raf']['accuracy_overall'],'raf_f1':chosen['raf']['macro_f1'],'eligible':chosen['eligible'],'best_epoch':best['epoch'],'peak_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
        if epoch>=5 and stale>=3:break
    # Freeze one validation-selected candidate before reading any test labels.
    checkpoint=torch.load(OUT/'checkpoints/best.pt',map_location='cpu',weights_only=True);student.load_state_dict(checkpoint['state_dict']);student=student.cpu().to(memory_format=torch.contiguous_format).eval()
    torch.onnx.export(student,torch.randn(2,3,260,260),str(OUT/'expert.onnx'),input_names=['input'],output_names=['logits'],dynamic_axes={'input':{0:'batch_size'},'logits':{0:'batch_size'}},opset_version=17,dynamo=False)
    (OUT/'selected.json').write_text(json.dumps({'epoch':checkpoint['epoch'],'weight':checkpoint['weight'],'validation':best,'model_sha256':hashlib.sha256((OUT/'expert.onnx').read_bytes()).hexdigest(),'test_evaluated':False},indent=2))
    print('TRAINING COMPLETE; candidate frozen, tests not yet evaluated',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--batch',type=int,default=64);parser.add_argument('--accumulation',type=int,default=1);parser.add_argument('--workers',type=int,default=0 if sys.platform=='win32' else 2);parser.add_argument('--optimized',action='store_true');parser.add_argument('--gpu-preprocess',action='store_true');parser.add_argument('--resume-weights',type=Path)
    with threadpool_limits(limits=2):main(parser.parse_args())
