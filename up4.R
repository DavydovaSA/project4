# Esly russkie bukvy ne otobrajautsa: File -> Reopen with encoding... UTF-8

# Используйте UTF-8 как кодировку по умолчанию!
# Установить кодировку в RStudio: Tools -> Global Options -> General, 
#  Default text encoding: UTF-8

# Математическое моделирование: Практика 4

library('ISLR')
library('GGally')
library('MASS')
library('mlbench')
data(BreastCancer)

?Titanic

my.seed <- 12345
train.percent <- 0.85

# Данные ---------------------------------------------------------------
data(Titanic)
Tit <- as.data.frame(Titanic)

# графики разброса
ggpairs(Tit)

# обучающая выборка
set.seed(my.seed)
inTrain <- sample(seq_along(Tit$Survived),
                  nrow(Tit) * train.percent)
df <- Tit[inTrain, ]
head(df)

# фактические значения на обучающей выборке
Факт <- df$Survived

# Логистическая регрессия ======================================================
model.logit <- glm(Survived ~ ., data = df,
                   family = 'binomial')
summary(model.logit)
# прогноз: вероятности принадлежности классу 'Yes'
p.logit <- predict(model.logit, df, 
                   type = 'response')
p.logit
Прогноз <- factor(ifelse(p.logit > 0.5,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('No', 'Yes'))
Прогноз
head(cbind(Факт, p.logit, Прогноз))

# матрица неточностей
table(Факт, Прогноз)

# LDA ==========================================================================
model.lda <- lda(Survived ~ ., data = df)
model.lda

# прогноз: вероятности принадлежности классу 'Yes' 
p.lda <- predict(model.lda, df, type = 'response')
head(p.lda$posterior)
Прогноз <- factor(ifelse(p.lda$posterior[, 'Yes'] > 0.5,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('No', 'Yes'))
Прогноз
# матрица неточностей
table(Факт, Прогноз)

# ROC-кривая для LDA ===========================================================

# считаем 1-SPC и TPR для всех вариантов границы отсечения
x <- NULL    # для (1 - SPC)
y <- NULL    # для TPR
x1 <- NULL    # для (1 - SPC)
y1 <- NULL    # для TPR

# заготовка под матрицу неточностей
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl) <- c('fact.No', 'fact.Yes')
colnames(tbl) <- c('predict.No', 'predict.Yes')
tbl
tbl1 <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl1) <- c('fact.No', 'fact.Yes')
colnames(tbl1) <- c('predict.No', 'predict.Yes')

# цикл по вероятностям отсечения
for (p in seq(0, 1, length = 500)){
Прогноз1 <- factor(ifelse(p.logit > p,
                         2, 1),
                  levels = c(1, 2),
                  labels = c('No', 'Yes'))
df.compare <- data.frame(Факт = Факт, 
                             Прогноз1 = Прогноз1)

# заполняем матрицу неточностей
tbl1[1, 1] <- nrow(df.compare[df.compare$Факт == 'No' & 
                               df.compare$Прогноз1 == 'No', ]) 
tbl1[2, 2] <- nrow(df.compare[df.compare$Факт == 'Yes' & 
                               df.compare$Прогноз1 == 'Yes', ])
tbl1[1, 2] <- nrow(df.compare[df.compare$Факт == 'No' & 
                               df.compare$Прогноз1 == 'Yes', ])
tbl1[2, 1] <- nrow(df.compare[df.compare$Факт == 'Yes' & 
                               df.compare$Прогноз1 == 'No', ])

# считаем характеристики
TPR1 <- tbl1[2, 2] / sum(tbl1[2, 2] + tbl1[2, 1])
y1 <- c(y1, TPR1)
SPC1 <- tbl1[1, 1] / sum(tbl1[1, 1] + tbl1[1, 2])
x1 <- c(x1, 1 - SPC1)
}

plot(x1, y1, 
          type = 'l', col = 'blue', lwd = 3,
          xlab = '(1 - SPC)', ylab = 'TPR', 
          xlim = c(0, 1), ylim = c(0, 1))


# цикл по вероятностям отсечения
for (p in seq(0, 1, length = 500)){
    # прогноз
    Прогноз <- factor(ifelse(p.lda$posterior[, 'Yes'] > p,
                             2, 1),
                      levels = c(1, 2),
                      labels = c('No', 'Yes'))
    
    # фрейм со сравнением факта и прогноза
    df.compare <- data.frame(Факт = Факт, 
                                 Прогноз = Прогноз)
    
    # заполняем матрицу неточностей
    tbl[1, 1] <- nrow(df.compare[df.compare$Факт == 'No' & 
                                   df.compare$Прогноз == 'No', ]) 
    tbl[2, 2] <- nrow(df.compare[df.compare$Факт == 'Yes' & 
                                   df.compare$Прогноз == 'Yes', ])
    tbl[1, 2] <- nrow(df.compare[df.compare$Факт == 'No' & 
                                   df.compare$Прогноз == 'Yes', ])
    tbl[2, 1] <- nrow(df.compare[df.compare$Факт == 'Yes' & 
                                   df.compare$Прогноз == 'No', ])
    
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

dev.off()
