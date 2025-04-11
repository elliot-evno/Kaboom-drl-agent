## I developed this Kaboom agent to learn more about Deep Q-learning.

**Improvements:** Implement a computer vision model that recognizes falling elements and the basket and then make the corresponding moves (up, down, left, right) into keypresses and train it using that. This is so that you can apply it to pretty much any Kaboom game out there. (I actually developed this because my lilbro wanted to cheat in this swedish game called Flaire where you win money by playing games like Kaboom).

To run: 
```bash
git clone https://github.com/elliotnn/kaboom-drl-agent.git
cd kaboom-drl-agent
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
python main.py
