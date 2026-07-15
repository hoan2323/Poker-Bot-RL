# Poker Bot RL – Texas Hold'em rút gọn

Đây là dự án học tăng cường dành cho sinh viên, mô phỏng poker hai người bằng bộ bài nhỏ và huấn luyện bot bằng Q-Learning. Project có chương trình train, evaluate, tạo biểu đồ, bộ test pytest và giao diện Streamlit.

README này là hướng dẫn chạy chính. `project.md` ở thư mục cha là đặc tả của giai đoạn cũ (hai vòng cược, chưa có training) và không còn mô tả chính xác toàn bộ code hiện tại.

## Luật đang dùng

- Hai người chơi: Player 0 và Player 1.
- Bộ bài 20 lá gồm `10, J, Q, K, A` của bốn chất; card id nằm trong `0..19`.
- Mỗi người nhận hai lá riêng. Board lần lượt mở flop ba lá, turn một lá và river một lá.
- Pot ban đầu là 2, tương ứng mỗi người đóng một ante.
- Action: `0 = check/call`, `1 = bet/raise`, `2 = fold`.
- Mỗi street cho tối đa một lần raise để hand luôn kết thúc hữu hạn.
- Reward được tính theo Player 0: thắng `+pot`, thua `-pot`, hòa `0`. `winner` luôn là `0`, `1` hoặc `None`.
- Project dùng thứ tự poker chuẩn: High Card < Pair < Two Pair < Three of a Kind < Straight < Flush < Full House < Four of a Kind < Straight Flush.

Vì mini deck chỉ có đúng năm rank, một flush năm lá hợp lệ luôn chứa đủ năm rank và đồng thời là straight flush. Code evaluator vẫn kiểm tra straight và flush trên cùng đúng năm lá.

## Cấu trúc

```text
environment.py          Môi trường, luật cược và hand evaluator
q_learning_agent.py     Q-table, state abstraction và chọn action
train.py                Vòng lặp huấn luyện
evaluate.py             Baseline và báo cáo đánh giá
game_results.py         Giao diện kết quả terminal ổn định cho caller
opponent_model.py       Phân loại hành vi đối thủ
training_artifacts.py   Metadata/version của artifact
plot.py                 Biểu đồ training và evaluation
app.py                  Giao diện Streamlit
test_*.py               Bộ test pytest
```

## Q-Learning và opponent baseline

Observation 186 chiều được nén thành state key gồm sức mạnh bài, draw, pot, street, trạng thái cược và profile đối thủ. Agent dùng epsilon-greedy, cập nhật Q-value bằng reward terminal kết hợp reward shaping nhỏ.

Training lấy mẫu các đối thủ:

- `heuristic`: đánh theo sức mạnh bài;
- `call_station`: ưu tiên check/call;
- `always_bet`: chủ động bet/raise khi hợp lệ;
- `random`: chọn ngẫu nhiên trong valid actions.

## Cài đặt

Khuyến nghị Python 3.10 trở lên và virtual environment:

```powershell
cd "C:\REL301m\code\poker\Poker-Bot-RL\poker-bot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Test

```powershell
python -m pytest -v
```

Test bao phủ environment, hand ranking/kicker, winner/reward, giới hạn raise, 500 episode ngẫu nhiên, Q-agent và evaluation.

## Train

Full training mặc định:

```powershell
python train.py
```

Smoke test nhanh và lưu sang thư mục riêng:

```powershell
python train.py --episodes 1000 --eval-interval 100 --output-dir smoke-artifacts
```

## Evaluate, biểu đồ và UI

```powershell
python evaluate.py
python plot.py
streamlit run app.py
```

`evaluate.py` chỉ dùng Q-table có metadata tương thích với environment hiện tại. Nếu artifact cũ, chương trình sẽ hướng dẫn train lại.

## Bot Arena: Đăng poker Bot vs Poker-Bot-RL Bot

Matchup mặc định trong **🤖 Bot Arena** là hai bot của hai project:

- **Đăng poker Bot** dùng **NFSP / Deep RL**, gồm DQN best-response và average policy network, được lưu bằng PyTorch checkpoint `.pt` trong folder `C:\REL301m\code\poker\Đăng poker`.
- **Poker-Bot-RL Bot** dùng **Tabular Q-Learning** với Q-table `.npy` trong folder main. Arena ưu tiên `artifacts/current/q_table.npy`, sau đó dùng `q_table.npy` nếu artifact hiện tại chưa tồn tại.

Hai adapter đều thi đấu trên cùng `ShortDeckPokerEnv` của branch main. Arena không copy hoặc import `environment.py` từ folder Đăng poker, nên luật game, state và valid actions của hai phía luôn giống nhau.

**Random Bot** và **Heuristic Bot** vẫn có trong selectbox nhưng chỉ phục vụ kiểm thử và phải được người dùng chủ động chọn. Khi Đăng poker Bot hoặc Poker-Bot-RL Bot thiếu/lỗi model, Start bị khóa và UI hiển thị hướng dẫn; hệ thống không âm thầm thay bot lỗi bằng Random/Heuristic.

Để chạy matchup chính, cần có:

- checkpoint `C:\REL301m\code\poker\Đăng poker\nfsp_agent_final.pt` cho Đăng poker Bot;
- Q-table tương thích tại `artifacts/current/q_table.npy` hoặc `q_table.npy` cho Poker-Bot-RL Bot.

## Artifact

- `q_table.npy`: Q-table đã học.
- `rewards.npy`: reward của từng episode.
- `win_rates.npy`, `draw_rates.npy`, `loss_rates.npy`: tỷ lệ tích lũy.
- `training_metadata.json`: version format, số episode và số state.
- `evaluation_summary.json`: kết quả các matchup.
- `evaluation_report.png` và `training_curve.png`: biểu đồ sinh bởi `plot.py`.

Các artifact được tạo trước khi sửa winner, showdown reward và evaluator không còn tương thích về ngữ nghĩa. Hãy train lại trước khi dùng để đánh giá hoặc demo kết quả chính thức.

## Hạn chế

- Đây là mini deck phục vụ học tập, không phải full 52-card Texas Hold'em.
- Bet size cố định, không mô phỏng stack, blind, all-in hoặc side pot.
- Q-agent chứa một số heuristic hỗ trợ ngoài Q-value thuần túy.
- `app.py` còn dài và chưa tách component.
- Hand evaluator hiện nằm chung trong `environment.py`.

## Future Work

- Tách `hand_evaluator.py` và `config.py`.
- Refactor `app.py` thành các component nhỏ.
- Thử ranking short-deck tùy biến, ví dụ flush cao hơn full house.
- Thử NFSP/deep RL dựa trên ý tưởng ở `dang_branch`, sau khi bổ sung action masking, observation đúng theo player và transition đúng lượt.
- Bổ sung stack, blind, bet sizing và nhiều luật poker thực tế hơn.
