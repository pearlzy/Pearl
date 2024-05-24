import torch
import json
import os
import tqdm
class Trainer(object):
    def __init__(self, model, pg, optimizer, args, dataset, config, distribution=None):

        self.model = model.cuda()
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.distribution = distribution
        self.data = dataset.data
        self.config = config
    def train_epoch(self, dataloader, ntriple, args):

        self.model.train()

        self.optimizer.zero_grad()
        total_loss = 0.0
        total_reward = 0.0
        counter = 0

        with tqdm.tqdm(total=ntriple, desc='Training',  unit='it', leave=True) as bar:
            bar.set_description('Train')
            for input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, rels, answers_single, answers_arr,  answers_type in dataloader:
                if self.args.cuda:
                    if type(input_ids) != list:
                        input_ids = input_ids.cuda()
                    if type(attention_mask) != list:
                        attention_mask = attention_mask.cuda()
                    if type(entity_time_ids_padded) != list:
                        entity_time_ids_padded = entity_time_ids_padded.cuda()
                    if type(entity_mask_padded) != list:
                        entity_mask_padded = entity_mask_padded.cuda()
                    if type(heads) != list:
                        heads = heads.cuda()
                    if type(tails) != list:
                        tails = tails.cuda()
                    if type(times) != list:
                        times = times.cuda()
                    if type(start_times) != list:
                        start_times = start_times.cuda()
                    if type(end_times) != list:
                        end_times = end_times.cuda()
                    if type(tails2) != list:
                        tails2 = tails2.cuda()
                    if type(types) != list:
                        types = types.cuda()
                    if type(rels) != list:
                        rels = rels.cuda()
                    if type(answers_single) != list:
                        answers_single = answers_single.cuda()
                    if type(answers_arr) != list:
                        answers_arr = answers_arr.cuda()
                    if type(answers_type) != list:
                        answers_type = answers_type.cuda()

                all_loss, all_logits,  all_actions_idx, current_entities, cts1, cts2, all_actions_scores = self.model(
                        input_ids, attention_mask,
                        entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2,
                        types, rels, answers_single, answers_arr, answers_type
                )
                reward = self.pg.get_reward(current_entities, cts1, cts2, answers_arr)
                if self.args.reward_shaping:
                    # reward shaping
                    dt1 = torch.abs(start_times - cts1)
                    dt2 = torch.abs(end_times - cts2)
                    dt = torch.abs(dt2+dt1)/2
                    delta_time = torch.ceil(dt).to(torch.int)
                    p_dt = []
                    for i in range(rels.shape[0]):
                        rel = rels[i].item()
                        dt = delta_time[i].item()
                        p_dt.append(self.distribution(rel, dt // self.args.time_span))
                    p_dt = torch.tensor(p_dt)
                    # cum_discounted_reward
                    if self.args.cuda:
                        p_dt = p_dt.cuda()
                    shaped_reward = (1 + p_dt) * reward
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward)
                else:
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)

                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)

                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1
                reinfore_loss.backward()
                if self.args.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                self.optimizer.step()
                total_loss += reinfore_loss
                total_reward += torch.mean(reward)
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(reinfore_loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item())

        return total_loss / counter, total_reward / counter


    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""

        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },
            os.path.join(self.args.save_path, checkpoint_path)
        )
