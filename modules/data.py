import numpy as np
import torch
from torch.autograd import Variable


class SessionDataLoader:
    def __init__(self, df, session_key, item_key, time_key, hidden, batch_size=50, training=True,
                 time_sort=False):
        """
        A class for creating session-parallel mini-batches.

        Args:
             df (pd.DataFrame): the dataframe to generate the batches from
             session_key (str): session ID
             item_key (str): item ID
             time_key (str): time ID
             batch_size (int): size of the batch
             hidden (torch.FloatTensor): initial hidden state(should be fed from the outside)
             training (bool): whether to generate the batches in the training mode. If False, Variables will be created with the flag `volatile = True`.
             time_sort (bool): whether to sort the sessions by time when generating batches
        """
        self.df = df

        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        self.hidden = hidden

        self.batch_size = batch_size
        self.training = training
        self.time_sort = time_sort

    def generate_batch(self):
        """ A generator function for producing session-parallel training mini-batches.

        Returns:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            hidden: previous hidden state
        """

        # initializations
        df = self.df
        click_offsets = self.get_click_offsets(df)  # 每个session开始位置 wrt. df 数组
        session_idx_arr = self.order_session_idx(df)  # seesion idx 数组

        iters = np.arange(self.batch_size)  # 保存当前Batch中的session id
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]  # 当前B个session的开始位置
        end = click_offsets[session_idx_arr[iters] + 1]
        finished = False

        while not finished:
            minlen = (end - start).min()  # 找到前B个session中最短的长度
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]  # start位置对应的item
            for i in range(minlen - 1):             # 用start迭代更新 idx_target idx_input (最短session用到倒数第二个item)
                # Build inputs, targets, and hidden states
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                if self.training:
                    input = Variable(torch.LongTensor(idx_input))  # (B)
                    target = Variable(torch.LongTensor(idx_target))  # (B)
                    hidden = Variable(self.hidden)
                else:
                    input = Variable(torch.LongTensor(idx_input), volatile=True) 
                    target = Variable(torch.LongTensor(idx_target), volatile=True)  # (B)
                    hidden = Variable(self.hidden, volatile=True)

                yield input, target, hidden
                #######################################################################################################
                # WARNING: after this step, hidden states should be updated from the outside
                # using `update_hidden(hidden.data)`
                #######################################################################################################

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)  # 对齐到最短session的最后item位置
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]  # 到达末尾的session. 更换新的session到mask的位置
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

            # reset the rnn hidden state to zero after transition
            if len(mask) != 0:
                self.hidden[:, mask, :] = 0

    def update_hidden(self, hidden):
        """ Update the hidden state from the outside """
        self.hidden = hidden

    def get_click_offsets(self, df):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """

        session_key = self.session_key
        offsets = np.zeros(df[session_key].nunique() + 1, dtype=np.int32)  # （|session| + 1）
        # group & sort the df by session_key and get the offset values
        offsets[1:] = df.groupby(session_key).size().cumsum()  # 每个session长度依次累加的数组

        return offsets

    def order_session_idx(self, df):
        """ Order the session indices """

        session_key = self.session_key
        time_key = self.time_key
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = df.groupby(session_key)[time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(df[session_key].nunique())

        return session_idx_arr