class StockEnvironment:
    """
    OPD算法专用量化RL环境:
        - Environment视为数据集内股票价格变化活动,可包含多支股票
        - 每个Episode是一段连续的K线信息
        - Episode本质上是对Environment的采样,Episode之间无关联,可在[时段, 股票]空间自由跳跃采样
        - Reward是Episode中每次行动对均价的优势
    """

    def __init__(self, stock_dict, start_time, end_time, episode_len, fully_obs, alpha=0.001):
        """
        从多支股票数据中采样Episode: 随机在[时段, 股票]空间采样Episode
        读取原始数据并包装为Episode
        :param stock_dict: Environment中所有股票的[name-id]字典
        :param start_time: Environment中所有股票K线的统一起始时间
        :param end_time: Environment中所有股票K线的统一结束时间
        :param episode_len: Environment中统一的Episode长度
        :param fully_obs: 标明本环境的可见性(行情对student agent部分可见)
        :param alpha: Reward中模拟交易扰动的惩罚系数
        """
        self.stock_dict = stock_dict
        self.start_time = start_time
        self.end_time = end_time
        self.episode_len = episode_len
        self.fully_obs = fully_obs
        self.episode = None
        self.kline_data_shelf = self._load_data()  # 读取数据集到内存(便于训练中快速采样)
        self.kline_len = min([kline_data.shape[0] for kline_data in self.kline_data_shelf])  # 数据集中股票行情的统一长度
        self.alpha = alpha
        assert np.std(np.array([kline_data.shape[0] for kline_data in self.kline_data_shelf])) < 5.0  # todo: 修改: 保证起始时间一致

    def _load_data(self):
        # 读取全部CSV-KLine数据到内存,约1.2G
        kline_data_shelf = []
        for stock_name in self.stock_dict.values():
            csv_file_path = os.path.join(RAW_CSV_PATH, stock_name)
            csv_file_pathname = os.path.join(csv_file_path, f"{stock_name}_combine_kline.csv")
            df = pd.read_csv(csv_file_pathname)
            kline_data_shelf.append(df)
        return kline_data_shelf

    def _public_obs_normalize(self, obs_public):
        assert len(obs_public.shape) == 2, "!! OBS正则化: 错误的OBS形状."
        normalized_obs = torch.cat((obs_public[:, :4] / obs_public[0, 0], (obs_public[:, 4] / obs_public[0, 4]).unsqueeze(1)), 1)
        return normalized_obs

    def _get_observation(self):
        obs_public = torch.tensor(np.array(self.episode.kline.iloc[:self.episode_len])).float()
        obs_private = torch.tensor(np.array([list(range(self.episode.t + 1)) +
                                             list([0.0] * (self.episode_len - self.episode.t - 1)),
                                             self.episode.remain_r[: self.episode_len]])) \
            .transpose(0, 1).float()
        # import pdb; pdb.set_trace()
        # todo: 数据正则化
        obs_public = self._public_obs_normalize(obs_public)
        # obs_private = torch.tensor(np.array([self.episode.t, self.episode.remain_inv])).float()
        return (obs_public, obs_private)

    def reset(self, utter=False):
        # 从Episode空间中采样&初始化episode实例
        stock_id = random.randint(0, len(self.stock_dict) - 1)
        start_ind = random.randint(0, self.kline_len - self.episode_len - 1)

        stock_id = 9
        start_ind = 63
        # start_ind = 334530
        # start_ind = 896430

        # stock_id = 9
        # start_ind = random.randint(0, 10 - 1)

        episode_kline_raw = self.kline_data_shelf[stock_id][start_ind: start_ind + self.episode_len + 1]  # 多读取1个时刻数据
        self.episode = OPDEpisode(stock_id, start_ind, self.fully_obs, episode_kline_raw)
        if utter:
            print(f"  - reset env: stock_id:{stock_id}, start_ind:{start_ind}, "
                  f"kline_len:{self.kline_len}, ave_price:{self.episode.ave_price}")  # todo: debug恢复
        # 构建返回值
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action_infer):
        """
        计算t时刻的订单(来自t-1时刻)
        """
        # 读取Episode数据
        kline = self.episode.kline
        t = self.episode.t
        # 更新剩余资金
        remain_inv_entire = self.episode.init_inv - self.episode.trade_record['volume'].sum()
        # 整形交易量
        action = np.clip(action_infer, 0, remain_inv_entire)
        if t == self.episode_len - 1:
            action = remain_inv_entire  # 最后时刻全部交易
        # 计算Instant-Reward
        trade_ratio = action / self.episode.init_inv
        trade_advantage = (kline['Open'][t + 1] / self.episode.ave_price) - 1.0  # 使用下一时刻初始股价计算当前时刻交易决策的(相对)价值
        # expolore_reward = (max(kline['Open']) / self.episode.ave_price - 1) * (self.episode.t / self.episode_len) * 0.1
        # import pdb; pdb.set_trace()
        reward = trade_ratio * trade_advantage - self.alpha * action ** 2  # 公式(4)
        # reward += expolore_reward  # todo: 鼓励向后探索的内部奖励
        self.episode.reward_r[t] = reward  # 交易决策后立即获得奖励反馈
        # 纪录交易
        self.episode.trade_record['volume'][t + 1] = action  # 交易决策登记
        self.episode.remain_inv -= self.episode.trade_record['volume'][t]  # 剩余资本计算
        self.episode.remain_r[t] = self.episode.remain_inv  # 剩余资本顺序记录
        # 构建返回值(gym风格)
        observation = self._get_observation()
        done = t >= self.episode_len - 1
        truncated = False
        info = {"true action": action}
        # 推进Elapsed-Time
        self.episode.t += 1  # 订单执行阶段完成,t时刻数据更新,时刻步进,进入观察决策阶段
        return observation, reward, done, truncated, info

    def render(self, timing="after_step", style="plot", prefix="prefix", show=True, save=False):
        # 按OPD论文图3形式展示当前Episode状态
        # 按照render位置计算"实际已完成的时刻"t
        if timing == "after_step":
            t = self.episode.t - 1
        else:
            raise ValueError(f"OPDEnvironment.step: timing '{timing}' is not valid!")
        if style == "debug":
            # if self.episode.t == self.episode_len:
            print("\n")
            print("t:", self.episode.t)
            print("trade_record:", list(self.episode.trade_record['volume']))
            print("remain_r:", self.episode.remain_r)
            print("reward_r:", self.episode.reward_r)
            print("reward:", sum(self.episode.reward_r))
        elif style == "text":
            print("\n>>episode:")
            print(f"  stock_id: {self.episode.stock_id}")
            print(f"  start_time: {self.episode.start_time}")
            print(f"  fully_obs: {self.episode.fully_obs}")
            print(f"  init_inv: {self.episode.init_inv}")
            print(f"  remain_inv: {self.episode.remain_inv}")
            print(f"  t: {t}")
            # print(f"  kline: {self.episode.kline}")
            print(f"  ave_price: {self.episode.ave_price}")
        elif style == "plot":
            # unobservable region shadow
            kline = self.episode.kline
            # trading marks
            trade_condition = self.episode.trade_record['volume'] > 0.0
            trade_signal = np.where(trade_condition, kline['Open'], np.nan)
            add_plot = mpf.make_addplot(trade_signal, type='scatter', marker='x', color='b') if any(list(trade_condition)) else []
            # 保存图像
            save_path_name = f"../output/env_fig/{prefix}_plot_{self.episode.t}.png"
            # plot figure
            fig, ax_list = mpf.plot(kline, addplot=add_plot,
                                    type='ohlc', volume=True, show_nontrading=False,
                                    style='yahoo',
                                    figratio=(18, 10), figscale=0.9, tight_layout=False,
                                    title=f'\n{STOCK_DICT[self.episode.stock_id]} | {kline.index[0]}({self.episode.start_time})',
                                    xlabel=f'fin_t={t:02.0f} | remain_inv={self.episode.remain_r[t]:1.3f} | '
                                           f'action={self.episode.trade_record["volume"][t + 1]:1.3f} | reward={self.episode.reward_r[t]:1.6f} | total_r={sum(self.episode.reward_r):1.6f}',
                                    # 观察值t,交易结果t-1
                                    ylabel='Price', ylabel_lower='Volume\n(Shares)',
                                    xrotation=30,
                                    returnfig=True,
                                    vlines=dict(vlines=[kline.index[ind] for ind in range(self.episode.t, self.episode_len + 1)],
                                                linewidths=367 / self.episode_len, colors='gray', alpha=0.3),
                                    # todo: 增加Action文字
                                    # savefig=save_path_name,
                                    )
            # 标注交易量
            text_range = t + 2
            for ind in range(text_range):
                trade_value = self.episode.trade_record["volume"][ind]
                trade_value_str = f'{trade_value:0.3f}' if trade_value else ''
                y = kline.loc[kline.index[ind], 'Open']
                ax_list[0].text(ind, y, trade_value_str)  # 观察值t,交易结果t-1
            if save:
                mpl.savefig(save_path_name)
            if show:
                mpf.show()
            # mpl.close('all')
        else:
            raise ValueError(f"env render style error! '{style}' is not valid.")