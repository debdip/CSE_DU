ubuntu 16.04
for starting apache and mysql commands
sudo systemctl restart apache2.service
sudo systemctl restart mysql.service 

for wordpress or website permission 

sudo chown www-data.www-data wordpress/
sudo chown -Rf www-data.www-data /var/www/html/
