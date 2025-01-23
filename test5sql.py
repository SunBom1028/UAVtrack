import pymysql

db = pymysql.connect(host='localhost',
                     user='root',
                     password='1234',
                     database='testsql')

cursor = db.cursor()

cursor.execute("SELECT VERSION()")

# 使用 fetchone() 方法获取单条数据.
data = cursor.fetchone()

print("Database version : %s " % data)

# 关闭数据库连接
db.close()