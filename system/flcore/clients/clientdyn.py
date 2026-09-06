import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientDyn(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha

        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        
        # ===== for FPI-based compression =====
        self.prev_model_vector = None     # θ_k^{t-1}
        self.prev_grad_vector  = None     # ∇L_k(θ_k^{t-1})
        self.rho = args.rho               # Top-ρ ratio

        

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha/2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.global_model_vector is not None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)
            
        # ===== collect current model & gradient =====
        current_model_vector = model_parameter_vector(self.model).detach().clone()

        # ===== recompute full local gradient at θ_k^t =====
        self.model.zero_grad()
        trainloader = self.load_train_data()

        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)

            if self.global_model_vector is not None:
                v1 = model_parameter_vector(self.model)
                loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                loss -= torch.dot(v1, self.old_grad)

            loss = loss / len(trainloader)
            loss.backward()

        grad_list = []
        for p in self.model.parameters():
            grad_list.append(p.grad.detach().clone().view(-1))

        current_grad_vector = torch.cat(grad_list, dim=0)

        
        if self.prev_model_vector is not None and self.prev_grad_vector is not None:
            delta_theta = current_model_vector - self.prev_model_vector
            delta_grad  = current_grad_vector - self.prev_grad_vector

            # FPI: | (∇L_t − ∇L_{t-1}) ⊙ Δθ |
            fpi = torch.abs(delta_grad * delta_theta)
        else:
            fpi = None

        if fpi is not None:
            num_params = fpi.numel()
            k = max(1, int(self.rho * num_params))

            if k < num_params:
                _, topk_idx = torch.topk(fpi, k, largest=True)
                mask = torch.zeros_like(fpi)
                mask[topk_idx] = 1.0

                # sparsify update Δθ
                sparse_delta = (current_model_vector - self.global_model_vector) * mask

                # reconstruct uploaded model
                new_model_vector = self.global_model_vector + sparse_delta
                self._vector_to_model(new_model_vector)
                uploaded_model_vector = new_model_vector.detach().clone()
            else:
                uploaded_model_vector = current_model_vector.clone()

                
        if fpi is not None:
            self.prev_model_vector = uploaded_model_vector
        else:
            self.prev_model_vector = current_model_vector.clone()

        self.prev_grad_vector = current_grad_vector.clone()




        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model_vector = model_parameter_vector(model).detach().clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha/2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def _vector_to_model(self, vector):
        vector_to_model(self.model, vector)



def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)

def vector_to_model(model, vector):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vector[pointer:pointer + numel].view_as(p))
        pointer += numel
