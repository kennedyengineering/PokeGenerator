# Setup environment
GREEN='\033[0;32m'
echo -e "${GREEN}Creating virtual environment"
python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}Installing packages"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}Finished"
