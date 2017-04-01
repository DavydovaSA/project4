library('ISLR')
library('GGally')
library('MASS')

my.seed <- 12345
train.percent <- 0.85

fileURL <- 'https://raw.githubusercontent.com/bdemeshev/coursera_metrics/master/lab_07/titanic3.csv'
df <- read.csv(fileURL, row.names = 1)
df <- df[, -3]
df <- df[,-(7:9)]
df <- df[,-(8:10)]

for (i in 3:7)
{df[, i] <- as.numeric(df[, i])}
df[, 1] <- as.numeric(df[, 1])
df <- na.omit(df)


ggpairs(df)

# обучающая выборка
set.seed(my.seed)
inTrain <- sample(seq_along(df$survived),
                  nrow(df) * train.percent)
df <- df[inTrain,]
df.nt <- df[-inTrain,]
head(df)

# фактические значения на обучающей выборке
Факт <- df$survived

# Логистическая регрессия ======================================================
model.logit <- glm(survived ~ ., data = df,
                   family = 'binomial')
summary(model.logit)

# обучающая выборка:
p.logit <- predict(model.logit, df, 
                   type = 'response')
p.logit
Прогноз <- factor(ifelse(p.logit > 0.5,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('0', '1'))
Прогноз
head(cbind(Факт, p.logit, Прогноз))
# матрица неточностей
table(Факт, Прогноз)


# LDA ==========================================================================
model.lda <- lda(survived ~ ., data = df)
model.lda

# обучающая выборка:
p.lda <- predict(model.lda, df, type = 'response')
head(p.lda$posterior)
Прогноз <- factor(ifelse(p.lda$posterior[, '1'] > 0.5,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('0', '1'))

Прогноз
# матрица неточностей
Факт <- df$survived
table(Факт, Прогноз)


# ROC-кривые на обучающей выборке: ======================================================================================================================

# ROC-кривая для LDA ===========================================================

# считаем 1-SPC и TPR для всех вариантов границы отсечения
x <- NULL    # для (1 - SPC)
y <- NULL    # для TPR

# заготовка под матрицу неточностей
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl) <- c('fact0', 'fact1')
colnames(tbl) <- c('predict0', 'predict1')
tbl

# цикл по вероятностям отсечения
for (p in seq(0, 1, length = 500)){
  # прогноз
  Прогноз <- factor(ifelse(p.lda$posterior[, '1'] > p,
                           2, 1),
                    levels = c(1, 2),
                    labels = c('0', '1'))
  
  # фрейм со сравнением факта и прогноза
  df.compare <- data.frame(Факт = Факт, 
                               Прогноз = Прогноз)
  
  # заполняем матрицу неточностей
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '1', ])
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '1', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '0', ])
  
  # считаем характеристики
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1])
  y <- c(y, TPR)
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2])
  x <- c(x, 1 - SPC)
}

# строим ROC-кривую
par(mar = c(5, 5, 1, 1))
# кривая
plot(x, y, 
     type = 'l', col = 'blue', lwd = 3,
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1))
# прямая случайного классификатора
abline(a = 0, b = 1, lty = 3, lwd = 2)


# ROC-кривая для логистической регрессии ===========================================================

# считаем 1-SPC и TPR для всех вариантов границы отсечения
x2 <- NULL    # для (1 - SPC)
y2 <- NULL    # для TPR

# заготовка под матрицу неточностей
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl) <- c('fact0', 'fact1')
colnames(tbl) <- c('predict0', 'predict1')
tbl

# цикл по вероятностям отсечения
for (p in seq(0, 1, length = 500)){
  # прогноз
  Прогноз <- factor(ifelse(p.logit > p,
                           2, 1),
                    levels = c(1, 2),
                    labels = c('0', '1'))
  
  # фрейм со сравнением факта и прогноза
  df.compare <- data.frame(Факт = Факт, 
                               Прогноз = Прогноз)
  
  # заполняем матрицу неточностей
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '1', ])
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '1', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '0', ])
  
  # считаем характеристики
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1])
  y2 <- c(y2, TPR)
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2])
  x2 <- c(x2, 1 - SPC)
}

# кривые
lines(x2,y2,col="red",lwd = 2)

# ROC-кривые на тестовой выборке: ======================================================================================================================

Факт <- df.nt$survived
p.logit.nt <- predict(model.logit, df.nt, 
                      type = 'response')
p.logit.nt
Прогноз <- factor(ifelse(p.logit.nt > 0.5,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('0', '1'))
Прогноз
head(cbind(Факт, p.logit.nt, Прогноз))
# матрица неточностей

table(Факт, Прогноз)

p.lda.nt <- predict(model.lda, df.nt, type = 'response')
head(p.lda.nt$posterior)
Прогноз <- factor(ifelse(p.lda.nt$posterior[, '1'] > 0.5,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('0', '1'))

Прогноз
# матрица неточностей
table(Факт, Прогноз)


# ROC-кривая для LDA ===========================================================

# считаем 1-SPC и TPR для всех вариантов границы отсечения
x <- NULL    # для (1 - SPC)
y <- NULL    # для TPR

# заготовка под матрицу неточностей
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl) <- c('fact.0', 'fact.1')
colnames(tbl) <- c('predict.0', 'predict.1')
tbl

# цикл по вероятностям отсечения
for (p in seq(0, 1, length = 500)){
  # прогноз
  Прогноз <- factor(ifelse(p.lda.nt$posterior[, '1'] > p,
                           2, 1),
                    levels = c(1, 2),
                    labels = c('0', '1'))
  
  # фрейм со сравнением факта и прогноза
  df.compare <- data.frame(Факт = Факт, 
                               Прогноз = Прогноз)
  
  # заполняем матрицу неточностей
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '1', ])
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '1', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '0', ])
  
  # считаем характеристики
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1])
  y <- c(y, TPR)
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2])
  x <- c(x, 1 - SPC)
}

# строим ROC-кривую
par(mar = c(5, 5, 1, 1))
# кривая
plot(x, y, 
     type = 'l', col = 'blue', lwd = 3,
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1))
# прямая случайного классификатора
abline(a = 0, b = 1, lty = 3, lwd = 2)


# ROC-кривая для логистической регрессии ===========================================================

# считаем 1-SPC и TPR для всех вариантов границы отсечения
x2 <- NULL    # для (1 - SPC)
y2 <- NULL    # для TPR

# заготовка под матрицу неточностей
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl) <- c('fact.0', 'fact.1')
colnames(tbl) <- c('predict.0', 'predict.1')
tbl

# цикл по вероятностям отсечения
for (p in seq(0, 1, length = 500)){
  # прогноз
  Прогноз <- factor(ifelse(p.logit.nt > p,
                           2, 1),
                    levels = c(1, 2),
                    labels = c('0', '1'))
  
  # фрейм со сравнением факта и прогноза
  df.compare <- data.frame(Факт = Факт, 
                               Прогноз = Прогноз)
  
  # заполняем матрицу неточностей
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '1', ])
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & 
                                 df.compare$Прогноз == '1', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & 
                                 df.compare$Прогноз == '0', ])
  
  # считаем характеристики
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1])
  y2 <- c(y2, TPR)
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2])
  x2 <- c(x2, 1 - SPC)
}

# кривые
lines(x2,y2,col="red",lwd = 2)

# Сравнение качества с помощью ROC-кривых ===================================

# На основании построенных графиков, делаем вывод, что модель LDA лучше, 
# поскольку прощадь под ROC кривой на тестовой выборке у неё больше, чем у модели логистической регрессии.
